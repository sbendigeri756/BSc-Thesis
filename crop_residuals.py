import numpy as np
import re
import os
import argparse
from pathlib import Path
from astropy.io import fits

REFF_FACTOR = 0.84
PIXEL_SCALE = 0.2
FRAME_HALF  = 250

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_workers",  type=int, default=4)
    return parser.parse_args()

def parse_filename(stem):
    match_d = re.search(r'_dist([\d.]+)_', stem)
    if not match_d:
        match_d = re.search(r'([\d.]+)Mpc', stem, re.IGNORECASE)
    if not match_d:
        raise ValueError(f"Could not parse distance from: {stem}")

    match_reff = re.search(r'_reff(\d+)', stem) or re.search(r'_r(\d+)', stem)
    if not match_reff:
        raise ValueError(f"Could not parse reff from: {stem}")

    return float(match_d.group(1)), float(match_reff.group(1))

def crop_image(img, stem):
    try:
        d_mpc, reff_kpc = parse_filename(stem)
    except ValueError as e:
        print(f"WARNING: {e}, skipping")
        return img  # return uncropped

    reff_arcsec = (reff_kpc * 1e3) * 206265.0 / (d_mpc * 1e6)
    reff_px  = reff_arcsec / PIXEL_SCALE
    half = int(np.round(REFF_FACTOR * reff_px))

    cy, cx = img.shape[0] // 2, img.shape[1] // 2

    if half >= FRAME_HALF:
        return img  # full frame for close galaxies

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    return img[y0:y1, x0:x1]

if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache: {args.input_dir}")
    cache = np.load(args.input_dir, allow_pickle=True)
    images = cache["images"]
    log_distances = cache["log_distances"]
    paths = list(cache["paths"])

    print(f"Found {len(paths)} images")
    print(f"Sample path: {paths[0]}")
    print(f"Sample stem: {os.path.basename(paths[0])}")

    cropped_images = []
    skipped = 0
    for i, (img, path) in enumerate(zip(images, paths)):
        stem = os.path.splitext(os.path.basename(path))[0]
        cropped = crop_image(img.astype(np.float32), stem)
        cropped_images.append(cropped)
        if i % 1000 == 0:
            print(f"  {i}/{len(paths)} — {stem} → {cropped.shape}")

    print("Saving cropped cache...")
    out_path = output_dir / "residuals_cropped.npz"
    np.savez_compressed(
        out_path,
        images=np.array(cropped_images, dtype=object), 
        log_distances=log_distances,
        paths=paths
    )
    print(f"Saved to {out_path}")
