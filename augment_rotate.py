#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from astropy.io import fits

#config block
INPUT_DIR = '/home5/s5511178/VIRGO01/Thesis/FynnGalaxiesTest/'
OUTPUT_DIR = '/scratch/s5511178/EllNoisy'

N_ROTATIONS = 4 #to rotate per 90 degs
N_WORKERS = 12

TARGET_MU0 = 20.0      #central galaxy SB (mag/arcsec²)
SKY_MU = 24.0          #Euclid-like H-band sky
ZPT = 27.0             #arbitrary effective zeropoint
PIXEL_SCALE = 0.2      #arcsec/pixel

def rotate_image(image, angle):

    angle = angle % 360

    if angle == 0:
        return image.copy()

    elif angle == 90:
        return np.rot90(image, k=1)

    elif angle == 180:
        return np.rot90(image, k=2)

    elif angle == 270:
        return np.rot90(image, k=3)

    else:
        raise ValueError(
            f"np.rot90 only supports multiples of 90 deg, got {angle}"
        )
#here convert such that central sb is 20 mag/arcsec^2
def calibrate_surface_brightness(image, target_mu0=20.0, zpt=27.0, pixel_scale=0.2):

    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2

    central_flux = np.mean(image[cy-1:cy+2, cx-1:cx+2])

    target_counts = (
        10**(-0.4 * (target_mu0 - zpt))
        * pixel_scale**2)

    scale_factor = target_counts / central_flux

    return image * scale_factor

#convert sky surf brightness into counts
def sky_counts_from_mu(sky_mu=24.0, zpt=27.0,pixel_scale=0.2): #gives counts/pixel
    res = (10**(-0.4 * (sky_mu - zpt)) * pixel_scale**2)
    return res

def add_poisson_noise(image_counts, sky_counts, rng=None):
    #constant sky background

    if rng is None:
        rng = np.random.default_rng()

    signal = image_counts + sky_counts

    noisy = rng.poisson(signal)

    return noisy.astype(np.float32)


def augment_image(image, angle, target_mu0=20.0, sky_mu=24.0, zpt=27.0, pixel_scale=0.2, rng=None):
    #rotate FIRST
    rotated = rotate_image(image, angle)

    calibrated = calibrate_surface_brightness(rotated, target_mu0=target_mu0, zpt=zpt, pixel_scale=pixel_scale)

    sky_counts = sky_counts_from_mu(sky_mu=sky_mu, zpt=zpt, pixel_scale=pixel_scale)

    noisy = add_poisson_noise(calibrated, sky_counts, rng=rng)

    return noisy

angles = [0, 90, 180, 270]

def augment_one_galaxy(args):

    (
        fits_path,
        output_dir,
        angles,
        target_mu0,
        sky_mu,
        zpt,
        pixel_scale
    ) = args

    rng = np.random.default_rng()

    try:

        with fits.open(fits_path) as hdul:

            image = hdul[0].data.astype(np.float32)

            header = hdul[0].header

        stem = fits_path.stem

        for i, angle in enumerate(angles):

            rotated = rotate_image(
                image,
                angle
            )


            calibrated = calibrate_surface_brightness(
                rotated,
                target_mu0=target_mu0,
                zpt=zpt,
                pixel_scale=pixel_scale
            )

            sky_counts = sky_counts_from_mu(
                sky_mu=sky_mu,
                zpt=zpt,
                pixel_scale=pixel_scale
            )

            noisy = add_poisson_noise(
                calibrated,
                sky_counts,
                rng=rng
            )

            outname = (
                f"{stem}_aug{i:02d}_rot{angle:03d}.fits"
            )

            outfile = output_dir / outname

            fits.writeto(
                outfile,
                noisy,
                header=header,
                overwrite=True
            )

        return 1

    except Exception as e:

        print(f"ERROR processing {fits_path}: {e}")

        return 0


#main (needed for GPU)
def parse_args():
    p = argparse.ArgumentParser(
        description="Augment galaxy FITS images with N evenly-spaced rotations and noise."
    )
    p.add_argument(
        "--input_dir",
        default=os.environ.get("AUG_INPUT_DIR", str(INPUT_DIR)),
        help="Directory containing input .fits files "
             "(or set AUG_INPUT_DIR env var).",
    )
    p.add_argument(
        "--output_dir",
        default=os.environ.get("AUG_OUTPUT_DIR", str(OUTPUT_DIR)),
        help="Directory to write augmented .fits files "
             "(or set AUG_OUTPUT_DIR env var).",
    )
    p.add_argument(
        "--n_rotations",
        type=int,
        default=N_ROTATIONS,
        help="Number of rotations per galaxy (default: 10, giving 36 deg steps).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NSLOTS", N_WORKERS)),
        help="Parallel worker processes (default: NSLOTS env var or cpu_count-1).",
    )

    p.add_argument( #I use this dry run to test how long it will take so that I don't overload the system by just running it
        "--dry_run",
        action="store_true",
        default=False,
        help="Print what would be written without actually writing.",
    )

    p.add_argument(
    "--pixel_scale",
    type=float,
    default=0.2,
    help="Pixel scale [arcsec/pixel]"
    )

    p.add_argument(
        "--target_mu",
        type=float,
        default=20.0,
        help="Target central surface brightness [mag/arcsec^2]"
    )

    p.add_argument(
        "--sky_brightness",
        type=float,
        default=24.0,
        help="Sky brightness [mag/arcsec^2]"
    )

    return p.parse_args()


def main():

    args = parse_args()

    input_dir = Path(args.input_dir)

    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fits_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".fits", ".fit", ".fts"}
    )

    if not fits_files:

        print(f"WARNING: No FITS files found in {input_dir}")

        return

    angles = [0, 90, 180, 270]

    target_mu0 = args.target_mu

    sky_mu = args.sky_brightness

    zpt = ZPT

    pixel_scale = args.pixel_scale

    print(f"Found {len(fits_files)} FITS files in {input_dir}")

    print(
        f"Rotation angles ({len(angles)} per galaxy): "
        + ", ".join(f"{a:.1f}°" for a in angles)
    )

    print(
        f"Expected output: "
        f"{len(fits_files)} x {len(angles)} "
        f"= {len(fits_files) * len(angles)} augmented files"
    )

    print(
        f"Workers: {args.workers} | "
        f"Output dir: {output_dir}"
    )

    if args.dry_run:

        print("DRY RUN — no files will be written.")

        return

    worker_args = [

        (
            f,
            output_dir,
            angles,
            target_mu0,
            sky_mu,
            zpt,
            pixel_scale
        )

        for f in fits_files

    ]

    if args.workers == 1:

        results = [
            augment_one_galaxy(a)
            for a in worker_args
        ]

    else:

        with Pool(processes=args.workers) as pool:

            results = pool.map(
                augment_one_galaxy,
                worker_args
            )

    n_success = sum(results)

    print(
        f"Done. Successfully processed "
        f"{n_success}/{len(fits_files)} galaxies."
    )

if __name__ == "__main__":
    main()
