import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models

#replicate algorithm
def replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            g = num_groups
            while num_channels % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, num_channels))
        else:
            replace_bn_with_gn(child, num_groups)

class SBFEfficientNetB0(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # Initialize without pretrained ImageNet weights since we are loading your checkpoint
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )
        replace_bn_with_gn(backbone)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

#info from filename
def normalise(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image) & (image != 0)]
    if len(finite) == 0:
        return image
    lo, hi = np.percentile(finite, 0.5), np.percentile(finite, 99.5)
    if hi > lo:
        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)
    return image

def parse_distance_from_filename(filename: str):
    """Fallback parser matching your regex patterns."""
    match = re.search(r"_d([\d.]+)_", filename)
    if match:
        return float(match.group(1))
    match = re.search(r"([\d.]+)Mpc", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"dist([\d.]+)", filename)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"Initializing exact custom SBFEfficientNetB0 structure...")
    model = SBFEfficientNetB0()
    
    print(f"Loading weights from key [model_state_dict] out of: {os.path.basename(args.ckpt_path)}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"Loading evaluation archive: {args.npz_path}")
    cache = np.load(args.npz_path, allow_pickle=True)
    images_raw = cache["images"]
    
    paths_key = None
    for k in ["paths", "filenames", "fnames", "meta"]:
        if k in cache:
            paths_key = k
            break
            
    results = []
    
    print("\nProcessing predictions (converting log10 outputs directly back to Mpc)...")
    for i in range(len(images_raw)):
        img = images_raw[i].astype(np.float32)
        if img.ndim == 3:
            img = img[0]
            
        img = normalise(img)
        
        #duplicate grayscale into 3 channels
        tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        
        fname = f"galaxy_idx_{i}"
        true_dist = None
        if paths_key is not None:
            path_item = cache[paths_key][i]
            fname = path_item.decode('utf-8') if isinstance(path_item, bytes) else str(path_item)
            true_dist = parse_distance_from_filename(fname)
            
        with torch.no_grad():
            pred_log = model(tensor).item()
            
        pred_dist_mpc = 10 ** pred_log #log back to physical
        
        print(f"  [{i+1}/{len(images_raw)}] {os.path.basename(fname)} -> True: {f'{true_dist:.1f}' if true_dist else 'N/A'} Mpc | Pred: {pred_dist_mpc:.1f} Mpc")
        
        results.append({
            "filename": fname,
            "true_distance_mpc": true_dist if true_dist else np.nan,
            "predicted_distance_mpc": pred_dist_mpc,
            "predicted_log10": pred_log
        })
        
    df = pd.DataFrame(results)
    csv_out = os.path.join(args.output_dir, "unseen_predictions.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nDone. Matrix outputs saved successfully to: {csv_out}")

if __name__ == "__main__":
    main()
