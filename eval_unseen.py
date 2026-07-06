#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from ml_alg_res_trial_7_0 import SBFEfficientNetB0, normalise


TITLE_FS = 16
LABEL_FS = 14
TICK_FS  = 12
ANNOT_FS = 10


def parse_dist_from_path(path):
    """Parse _dist<dist>_ from standard ArtPop filenames."""
    m = re.search(r"_dist([\d.]+)_", os.path.basename(str(path)))
    return float(m.group(1)) if m else None


def load_ensemble(ckpt_dir, max_folds, dropout, device):
    models = []
    for fold in range(1, max_folds + 1):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] fold {fold} not found: {ckpt_path}")
            continue
        # Use the EXACT same model class as training
        model = SBFEfficientNetB0(pretrained=False, dropout=dropout)
        ckpt  = torch.load(ckpt_path, map_location=device)
        # Checkpoint always has 'model_state_dict' key from our training script
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        epoch = ckpt.get("epoch", "?")
        val   = ckpt.get("val_loss", float("nan"))
        print(f"  Loaded fold {fold}  epoch={epoch}  val_loss={val:.6f}")
        models.append(model)
    print(f"  Ensemble: {len(models)} fold models")
    return models


@torch.no_grad()
def predict_single(models, image_np, device):
    img = normalise(image_np.astype(np.float32))
    tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    fold_preds = [model(tensor).item() for model in models]
    mean_log = float(np.mean(fold_preds))
    std_log  = float(np.std(fold_preds))
    mean_mpc = 10 ** mean_log
    std_mpc  = max(10 ** (mean_log + std_log) - mean_mpc, 0.0)
    return mean_mpc, std_mpc, fold_preds

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    print("\nLoading ensemble checkpoints...")
    models = load_ensemble(args.ckpt_dir, args.max_folds, args.dropout, device)
    if not models:
        raise RuntimeError("No checkpoints loaded. Check --ckpt_dir and --max_folds.")

    print(f"\nLoading NPZ: {args.npz}")
    data   = np.load(args.npz, allow_pickle=True)
    images = data["images"]
    print(f"  images shape : {images.shape}  dtype={images.dtype}")
    print(f"  NPZ keys     : {list(data.files)}")

    true_mpcs = []
    if "paths" in data.files:
        paths = list(data["paths"])
        for p in paths:
            d = parse_dist_from_path(p)
            true_mpcs.append(d)
        print(f"  Parsed distances from 'paths' key: "
              f"{sum(1 for d in true_mpcs if d is not None)}/{len(true_mpcs)} found")
    elif "log_distances" in data.files:
        true_mpcs = list(10 ** data["log_distances"].astype(np.float32))
        print(f"  Using 'log_distances' key: {len(true_mpcs)} distances")
    else:
        true_mpcs = [None] * len(images)
        print("  WARNING: no distance information found in NPZ")

    print(f"\nRunning inference on {len(images)} images...")
    pred_mpcs, std_mpcs, true_list = [], [], []

    for i in range(len(images)):
        img = images[i]
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if img.ndim == 3:
            img = img[0]

        pred, std, _ = predict_single(models, img, device)
        pred_mpcs.append(pred)
        std_mpcs.append(std)
        true_list.append(true_mpcs[i])

        t_str = f"{true_mpcs[i]:.1f}" if true_mpcs[i] is not None else "?"
        if i % 10 == 0 or i == len(images) - 1:
            print(f"  [{i+1:4d}/{len(images)}]  "
                  f"true={t_str:>7} Mpc  pred={pred:7.1f} Mpc  std={std:5.1f} Mpc  "
                  f"img={img.shape[0]}x{img.shape[1]}px")

    pred_mpcs = np.array(pred_mpcs)
    std_mpcs  = np.array(std_mpcs)

    labelled = [(t, p) for t, p in zip(true_list, pred_mpcs) if t is not None]
    if labelled:
        t_arr = np.array([x[0] for x in labelled])
        p_arr = np.array([x[1] for x in labelled])
        frac  = np.abs(p_arr - t_arr) / t_arr
        print(f"\n=== Metrics (N={len(labelled)}) ===")
        print(f"  MAE          = {mean_absolute_error(np.log10(t_arr), np.log10(p_arr)):.4f} dex")
        print(f"  MAE%         = {frac.mean()*100:.2f}%")
        print(f"  Median err   = {np.median(np.abs(p_arr-t_arr)):.2f} Mpc")
        print(f"  Mean err     = {np.mean(np.abs(p_arr-t_arr)):.2f} Mpc")
        print(f"  Within  5%%  = {(frac<=0.05).mean()*100:.1f}%")
        print(f"  Within 10%%  = {(frac<=0.10).mean()*100:.1f}%")
        try:
            print(f"  R²           = {r2_score(np.log10(t_arr), np.log10(p_arr)):.4f}")
        except Exception:
            pass
    else:
        print("\nNo true distances available — skipping metrics")
        t_arr, p_arr = None, None

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("filename,true_mpc,pred_mpc,std_mpc,error_mpc,frac_error_pct\n")
        paths_list = list(data["paths"]) if "paths" in data.files else \
                     [f"galaxy_{i}" for i in range(len(images))]
        for i, (path, pred, std, true) in enumerate(
                zip(paths_list, pred_mpcs, std_mpcs, true_list)):
            err  = pred - true if true is not None else ""
            frac = abs(pred-true)/true*100 if true is not None else ""
            err_str = f"{err:.2f}"  if isinstance(err,  float) else ""
            frac_str =  f"{frac:.1f}" if isinstance(frac, float) else ""
            f.write(f"{os.path.basename(str(path))},"
                    f"{true if true is not None else ''},"
                    f"{pred:.2f},{std:.2f},"
                    f"{err_str},"
                    f"{frac_str}\n")
    print(f"\nSaved CSV -> {csv_path}")

    if t_arr is not None and len(t_arr) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Unseen distance evaluation", fontsize=TITLE_FS+2, fontweight="bold")

        ax = axes[0]
        ax.errorbar(t_arr, p_arr,
                    yerr=std_mpcs[:len(t_arr)],
                    fmt="o", ms=6, color="steelblue", alpha=0.7,
                    ecolor="lightgrey", elinewidth=1, capsize=3,
                    label="Ensemble mean ± fold std")
        lims = [min(t_arr.min(), p_arr.min())*0.9,
                max(t_arr.max(), p_arr.max())*1.1]
        ax.plot(lims, lims, "r--", lw=1.5, label="1:1")
        ax.set_xlabel("True distance (Mpc)", fontsize=LABEL_FS)
        ax.set_ylabel("Predicted distance (Mpc)", fontsize=LABEL_FS)
        ax.set_title("Predicted vs true", fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=TICK_FS)
        ax.grid(True, alpha=0.4, linestyle="--")
        frac_all = np.abs(p_arr-t_arr)/t_arr
        txt = (f"N={len(t_arr)}\n"
               f"MAE%={frac_all.mean()*100:.1f}%\n"
               f"Within 5%: {(frac_all<=0.05).mean()*100:.1f}%\n"
               f"Within 10%: {(frac_all<=0.10).mean()*100:.1f}%")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=ANNOT_FS,
                va="top", bbox=dict(boxstyle="round", fc="white", ec="grey"))

        ax = axes[1]
        residuals = p_arr - t_arr
        ax.scatter(t_arr, residuals, s=30, color="steelblue", alpha=0.7)
        ax.axhline(0, color="k", lw=1.5, ls="--")
        d_range = np.linspace(lims[0], lims[1], 200)
        ax.fill_between(d_range, -0.05*d_range,  0.05*d_range,
                        alpha=0.15, color="green", label="±5%")
        ax.fill_between(d_range, -0.10*d_range,  0.10*d_range,
                        alpha=0.10, color="orange", label="±10%")
        ax.set_xlabel("True distance (Mpc)", fontsize=LABEL_FS)
        ax.set_ylabel("Predicted - True (Mpc)", fontsize=LABEL_FS)
        ax.set_title("Residuals", fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=TICK_FS)
        ax.grid(True, alpha=0.4, linestyle="--")

        plt.tight_layout()
        out_plot = os.path.join(args.out_dir, "predictions_1to1.png")
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot -> {out_plot}")

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",        required=True,
                        help="Path to cropped residuals NPZ")
    parser.add_argument("--ckpt_dir",   required=True,
                        help="Dir containing fold_1/best_model.pt etc.")
    parser.add_argument("--out_dir",    default="./unseen_eval")
    parser.add_argument("--max_folds",  type=int,   default=5)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--device",     type=str,   default="cuda")
    args = parser.parse_args()
    main(args)
