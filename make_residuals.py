# -*- coding: utf-8 -*-
import os
import re
import argparse
import multiprocessing as mp

import numpy as np
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from scipy.ndimage import gaussian_filter
from scipy.stats import sigmaclip

PIXEL_SCALE_ARCSEC = 0.2  
PSF_SIGMA_PX = 1.0  
IMAGE_SIZE = 501  

#we have to mask the places where the model is <1% of the peak, to avoid blowing up by dividing by 0
MODEL_MASK_FRACTION = 0.01 

#use filename
_FNAME_RE = re.compile(
    r"spiral"
    r"_dist(?P<dist>[0-9.]+)"
    r"_feh(?P<feh>[+-]?[0-9.]+)"
    r"_n(?P<n>[0-9.]+)"
    r"_s(?P<nstars>[-+0-9.eE]+)"
    r"_age(?P<age>[0-9.]+)"
    r"_e(?P<ellip>[0-9.]+)"
    r"_pa(?P<pa>[0-9.]+)"
    r"_r(?:eff)?(?P<reff>[0-9.]+)"
    r"_aug(?P<aug>[0-9]+)"
    r"_rot(?P<rot>[0-9]+)"
)

def parse_filename(fname):
    m = _FNAME_RE.search(os.path.basename(fname))
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname}")
    p = m.groupdict()
    return {
        "dist_mpc": float(p["dist"]),
        "feh": float(p["feh"]),
        "sersic_n": float(p["n"]),
        "nstars": float(p["nstars"]),
        "log_age": float(p["age"]),
        "ellip": float(p["ellip"]),
        "pa_orig": float(p["pa"]), #artpop deg pa
        "rot_angle": float(p["rot"]), #augmentation deg
        "reff_kpc": float(p["reff"]),
        "aug_idx": int(p["aug"]),
    }

def reff_kpc_to_pixels(reff_kpc, dist_mpc, pixel_scale_arcsec):
    #kpc to arcsec: angle_arcsec = (r_kpc/d_kpc) * (180/π * 3600)
    dist_kpc = dist_mpc*1e3
    reff_arcsec = (reff_kpc / dist_kpc) * (180.0 / np.pi) * 3600.0
    return reff_arcsec / pixel_scale_arcsec

#smooth sersic model
def make_sersic_model(params, image_size=IMAGE_SIZE,
                      pixel_scale=PIXEL_SCALE_ARCSEC,
                      psf_sigma=PSF_SIGMA_PX):
    cx = cy = (image_size - 1) / 2.0 #get central pixel

    reff_px = reff_kpc_to_pixels(params["reff_kpc"], params["dist_mpc"],
                                  pixel_scale)
    
    #PA= original PA + rotation, and it's symmetric by 180 degs
    pa_eff = (params["pa_orig"] + params["rot_angle"]) % 180.0
    theta_rad = np.deg2rad(pa_eff)

    y_idx, x_idx = np.mgrid[0:image_size, 0:image_size]

    sersic = Sersic2D(
        amplitude=1.0,
        r_eff=reff_px,
        n=params["sersic_n"],
        x_0=cx,
        y_0=cy,
        ellip=params["ellip"],
        theta=theta_rad,
    )

    model = sersic(x_idx, y_idx).astype(np.float64)
    model = np.clip(model, 0, None)
    model = gaussian_filter(model, sigma=psf_sigma) #convolve w gaussian

    #normalise to 1 sum
    total = model.sum()
    if total > 0:
        model /= total

    return model, reff_px

#divide image by smooth image
def compute_residual(image, model_scaled, mask_fraction=MODEL_MASK_FRACTION):
    threshold = mask_fraction * model_scaled.max()
    safe_model = model_scaled.copy()
    bad = safe_model < threshold
    safe_model[bad] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (image.astype(np.float64) - safe_model) / np.sqrt(safe_model)

    residual[bad] = np.nan
    return residual.astype(np.float32)

#normalise w sigma clipping as usual betw 0.5% and 99.5%
def normalise_residual(residual, lo_pct=0.5, hi_pct=99.5):
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return residual
    vmin = np.percentile(finite, lo_pct)
    vmax = np.percentile(finite, hi_pct)
    if vmax == vmin:
        return np.zeros_like(residual)
    out = (residual - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(residual)] = np.nan
    return out.astype(np.float32)

def process_file(args):
    fits_path, output_dir = args
    fname = os.path.basename(fits_path)
    out_path = os.path.join(output_dir, fname.replace(".fits", "_residual.fits"))

    if os.path.exists(out_path):
        print(f"[SKIP] {fname}", flush=True)
        return fname, True
    try:
        params = parse_filename(fname)
    except ValueError as e:
        print(f"[PARSE ERROR] {e}", flush=True)
        return fname, False
    try:
        with fits.open(fits_path) as hdul:
            image = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()

        model_unit, reff_px = make_sersic_model(params) #smooth model

        model_scaled = model_unit.astype(np.float32)

        residual = compute_residual(image, model_scaled) #dividing

        residual_norm = normalise_residual(residual) #normalise
        residual_norm = np.nan_to_num(residual_norm, nan=0.0) #make any NaN's into 0

        header["RESIDUAL"] = (True, "image/sersic_model residual")
        header["REFF_PX"] = (float(reff_px), "r_eff in pixels")
        header["PA_EFF"] = (float((params["pa_orig"] + params["rot_angle"]) % 180), "effective PA after rotation (deg)")
        hdu = fits.PrimaryHDU(data=residual_norm, header=header)
        hdu.writeto(out_path, overwrite=True)

        print(f"[DONE] {fname}  reff={reff_px:.1f}px", flush=True)
        return fname, True

    except Exception as e:
        print(f"[ERROR] {fname}: {e}", flush=True)
        return fname, False

#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate residual FITS images.")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing noisy pre-normalised FITS files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write residual FITS files.")
    parser.add_argument("--n_workers",  type=int, default=4,
                        help="Number of parallel worker processes.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fits_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".fits") and "_residual" not in f
    )
    print(f"Found {len(fits_files)} FITS files to process.", flush=True)

    tasks = [(f, args.output_dir) for f in fits_files]

    with mp.Pool(processes=args.n_workers) as pool:
        results = pool.map(process_file, tasks)

    n_ok = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_ok
    print(f"\nFinished: {n_ok} OK, {n_fail} failed", flush=True)
