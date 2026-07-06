#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
from astropy.modeling.models import Gaussian2D

rng = np.random.default_rng(50)

#get distance from filename
def parse_distance_from_filename(filepath):
    basename = os.path.basename(filepath)

    match = re.search(r"_dist([\d.]+)_", basename)
    if match:
        return float(match.group(1))

    match = re.search(r"_d([\d.]+)_", basename)
    if match:
        return float(match.group(1))

    match = re.search(r"([\d.]+)Mpc", basename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    raise ValueError(f"Could not determine distance from {basename}")


def add_point_sources(image, rng):

    contaminated = image.copy()

    ny, nx = contaminated.shape

    y, x = np.mgrid[:ny, :nx]

    #randomly choose if we want 1-3 stars
    n_sources = rng.integers(1, 4)

    for _ in range(n_sources):

        x0 = rng.uniform(10, nx - 10)
        y0 = rng.uniform(10, ny - 10)

        star = Gaussian2D(
            amplitude=0.5,
            x_mean=x0,
            y_mean=y0,
            x_stddev=1.0,
            y_stddev=1.0,
        )

        contaminated += star(x, y)

    contaminated = np.clip(contaminated, 0.0, 1.0)

    return contaminated, n_sources

#main
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        required=True,
        help="Input cropped residual dataset"
    )

    parser.add_argument(
        "--output_npz",
        required=True,
        help="Output robustness-test dataset"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading dataset...")

    data = np.load(args.input_npz, allow_pickle=True)

    images = data["images"]
    log_distances = data["log_distances"]
    paths = data["paths"]

    print(f"Loaded {len(images)} images.")

    #group indices w distane
    distance_groups = {}

    for i, path in enumerate(paths):

        dist = parse_distance_from_filename(path)

        if dist not in distance_groups:
            distance_groups[dist] = []

        distance_groups[dist].append(i)

    print("\nDistances found:")

    for dist in sorted(distance_groups):
        print(f"{dist:6.1f} Mpc : {len(distance_groups[dist])} images")

    #get two images per distance
    selected_indices = []

    print("\nSelecting images:")

    for dist in sorted(distance_groups):

        indices = np.array(distance_groups[dist])

        n_select = min(2, len(indices))

        chosen = rng.choice(indices, size=n_select, replace=False)

        selected_indices.extend(chosen.tolist())

        print(f"{dist:6.1f} Mpc -> selected {n_select}")

    selected_indices = np.array(selected_indices)

    print(f"\nSelected {len(selected_indices)} images.\n")

    #inject point sources
    contaminated_images = []
    n_sources_list = []
    for idx in selected_indices:

        img = np.asarray(images[idx], dtype=np.float32)
        contaminated, n_sources = add_point_sources(img, rng)

        contaminated_images.append(contaminated.astype(np.float32))
        n_sources_list.append(n_sources)

    print(f"\nProcessed {len(contaminated_images)} images.")

    contaminated_images = np.array(contaminated_images, dtype=object)

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)

    np.savez_compressed(
        args.output_npz,
        images=contaminated_images,
        log_distances=log_distances[selected_indices],
        paths=paths[selected_indices],
        n_point_sources=np.array(n_sources_list, dtype=np.int32),
    )

    print(f"\nSaved contaminated dataset to:")
    print(f"  {args.output_npz}")

    print("\nPoint-source summary:")
    unique, counts = np.unique(n_sources_list, return_counts=True)
    for n, c in zip(unique, counts):
        print(f"  {n} point source(s): {c} images")



if __name__ == "__main__":
    main()
