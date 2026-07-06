#!/usr/bin/env python3

import os
import re
import glob
import argparse
import numpy as np
from astropy.io import fits


def parse_distance_from_filename(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"_dist([\d.]+)_", basename)
    if match:
        return float(match.group(1))
    match = re.search(r"([\d.]+)Mpc", basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Could not find distance from filename: {basename}\n"
        "Expected '_d<distance>_' or '<distance>Mpc' in the filename."
    )


def validate_and_load(filepath):
    with fits.open(filepath) as hdul:
        image = hdul[0].data.astype(np.float32)

    if image is None:
        raise ValueError("Image data is None")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("Empty image")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(image).all():
        raise ValueError("Image still contains non-finite values after nan_to_num")
    if image.std() < 1e-8:
        raise ValueError(f"Near-zero std ({image.std():.2e}) – likely blank")
    if np.all(image == 0):
        raise ValueError("Image is entirely zero")

    return image

def main(args):
    fits_paths = sorted(
        glob.glob(os.path.join(args.data_dir, "**", "*.fits"), recursive=True)
    )
    if not fits_paths:
        raise FileNotFoundError(f"No .fits files found under {args.data_dir}")
    print(f"Found {len(fits_paths)} FITS files — processing...")

    images_list = []
    log_distances_list = []
    paths_list = []
    skipped = []

    for i, filepath in enumerate(fits_paths):
        if (i + 1) % 500 == 0 or (i + 1) == len(fits_paths):
            print(f"  [{i+1}/{len(fits_paths)}]  skipped so far: {len(skipped)}")

        try:
            distance = parse_distance_from_filename(filepath)
            image = validate_and_load(filepath)
        except Exception as e:
            msg = f"{filepath} | {type(e).__name__}: {e}"
            print(f"[WARNING] Skipping: {msg}")
            skipped.append(msg)
            continue

        images_list.append(image)
        log_distances_list.append(np.log10(distance))
        paths_list.append(filepath)

    if not images_list:
        raise RuntimeError("No valid images loaded — check your data directory.")

    images  = np.stack(images_list, axis=0)      
    log_distances = np.array(log_distances_list, dtype=np.float32)  
    paths = np.array(paths_list, dtype=object)    

    print(f"\nLoaded {len(images)} images  ({len(skipped)} skipped)")
    print(f"Image array shape: {images.shape}  dtype={images.dtype}")
    dists = 10**log_distances
    print(f"Distance range: {dists.min():.1f} – {dists.max():.1f} Mpc")
    print(f"log10(d) range: {log_distances.min():.3f} – {log_distances.max():.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        images=images,
        log_distances=log_distances,
        paths=paths,
    )
    print(f"\nSaved cache → {args.output}")
   
    if skipped:
         with open(args.skip_log, "w") as f:
             f.write("\n".join(skipped) + "\n")
         print(f"Skip log        → {args.skip_log}  ({len(skipped)} entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a directory of FITS images to a compressed .npz cache"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing FITS files (searched recursively)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the .npz cache file (e.g. /scratch/dataset_cache.npz)",
    )
    parser.add_argument(
        "--skip_log", type=str, default="skipped_files_preprocessing.txt",
        help="Text file to record any FITS files that failed validation",
    )
    args = parser.parse_args()
    main(args)
