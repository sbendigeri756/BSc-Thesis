#!/usr/bin/env python3

import os
import argparse
import numpy as np
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter, zoom
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.aperture import EllipticalAperture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SIM_PIXEL_SCALE  = 0.2    
SIM_PSF_FWHM     = 0.2   

SIM_PSF_FWHM_AS  = 1.0 * 2.355 * SIM_PIXEL_SCALE  # 0.471 arcsec

# HST WFC3/IR drizzled parameters
HST_PIXEL_SCALE  = 0.13   # arcsec/px
HST_PSF_FWHM_AS  = 0.15   # arcsec (approximate for F160W)


def normalise(image):
    finite = image[np.isfinite(image) & (image != 0)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo, hi = np.percentile(finite, 0.5), np.percentile(finite, 99.5)
    if hi <= lo:
        return np.zeros_like(image)
    out = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)

#didn't use this in the end
def psf_match(image, src_fwhm_as, tgt_fwhm_as, pixel_scale):
    src_sigma_px = (src_fwhm_as / 2.355) / pixel_scale
    tgt_sigma_px = (tgt_fwhm_as / 2.355) / pixel_scale
    if tgt_sigma_px <= src_sigma_px:
        print(f"  [PSF] Target PSF ({tgt_fwhm_as:.3f}\") narrower than "
              f"source ({src_fwhm_as:.3f}\") — no convolution applied.")
        return image
    kernel_sigma_px = np.sqrt(tgt_sigma_px**2 - src_sigma_px**2)
    print(f"  [PSF] Convolving: src={src_fwhm_as:.3f}\"  "
          f"tgt={tgt_fwhm_as:.3f}\"  kernel_sigma={kernel_sigma_px:.2f}px")
    return gaussian_filter(image, sigma=kernel_sigma_px)


def resample_area_average(image, src_scale, tgt_scale):
    factor = src_scale / tgt_scale   # >1 means we're making image coarser
    print(f"  [Resample] {src_scale}\"/px -> {tgt_scale}\"/px  "
          f"factor={factor:.3f}")
    if abs(factor - 1.0) < 0.01:
        print("  [Resample] Factor ~1 — skipping")
        return image

    if factor > 1.0:
        block = int(np.round(factor))
        if block < 2:
            print(f"  [Resample] Non-integer factor {factor:.3f} — using zoom order=1")
            return zoom(image, 1.0/factor, order=1).astype(np.float32)
        ny, nx = image.shape
        ny_trim = (ny // block) * block
        nx_trim = (nx // block) * block
        trimmed = image[:ny_trim, :nx_trim]
        resampled = trimmed.reshape(ny_trim//block, block,
                                    nx_trim//block, block).mean(axis=(1, 3))
        print(f"  [Resample] Block={block}  "
              f"{image.shape} -> {resampled.shape}")
        return resampled.astype(np.float32)
    else:
        return zoom(image, 1.0/factor, order=1).astype(np.float32)


def fit_ellipse_model(image, plot_path=None):
    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2

    sma_init = max(10, min(ny, nx) // 8)
    geometry = EllipseGeometry(
        x0=cx, y0=cy,
        sma=float(sma_init),
        eps=0.2,
        pa=0.0,
    )

    ellipse = Ellipse(image, geometry)

    print("  [Ellipse] Fitting isophotes...")
    try:
        isolist = ellipse.fit_image(
            sma0=sma_init,
            minsma=1.0,
            maxsma=min(ny, nx) / 2.0 * 0.9,
            step=0.1,
            nclip=3,
            maxgerr=1.0,
            fix_center=False,
        )
    except Exception as e:
        print(f"  [Ellipse WARNING] Full fit failed ({e}), retrying with fixed centre")
        geometry = EllipseGeometry(x0=cx, y0=cy,
                                   sma=float(sma_init), eps=0.2, pa=0.0)
        ellipse  = Ellipse(image, geometry)
        isolist  = ellipse.fit_image(
            sma0=sma_init,
            minsma=1.0,
            maxsma=min(ny, nx) / 2.0 * 0.9,
            step=0.1,
            nclip=3,
            maxgerr=1.0,
            fix_center=True,
        )

    print(f"  [Ellipse] Fit {len(isolist)} isophotes  "
          f"sma={isolist[0].sma:.1f}–{isolist[-1].sma:.1f} px")

    from photutils.isophote import build_ellipse_model
    model = build_ellipse_model(image.shape, isolist)
    model = np.clip(model, 0, None).astype(np.float64)

    if plot_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin, vmax = np.percentile(image[image > 0], [1, 99])
        axes[0].imshow(image,        cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax)
        axes[0].set_title("Original image", fontsize=12, fontweight="bold")
        axes[1].imshow(model,         cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax)
        axes[1].set_title("Ellipse model", fontsize=12, fontweight="bold")
        axes[2].imshow(image - model, cmap="RdBu_r",  origin="lower",
                       vmin=-vmax*0.1, vmax=vmax*0.1)
        axes[2].set_title("image − model", fontsize=12, fontweight="bold")
        for ax in axes:
            ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Ellipse] Saved diagnostic plot: {plot_path}")

    return model


def compute_residual(image, model, mask_fraction=0.01):
    """
    (image - model) / sqrt(model)

    Pixels where model < mask_fraction * model.max() are set to NaN
    (low-signal outskirts where division is unreliable), then set to 0
    for CNN compatibility.
    """
    threshold = mask_fraction * model.max()
    safe_model = model.copy()
    bad = safe_model < threshold
    safe_model[bad] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (image.astype(np.float64) - safe_model) / np.sqrt(safe_model)

    residual[bad] = np.nan
    return residual.astype(np.float32)


def subtract_background(image):
    """
    Simple sigma-clipped background subtraction using corner regions.
    For HST images the background is very low but non-zero.
    """
    ny, nx = image.shape
    margin = max(10, min(ny, nx) // 8)
    corners = np.concatenate([
        image[:margin,   :margin  ].ravel(),
        image[:margin,   -margin: ].ravel(),
        image[-margin:,  :margin  ].ravel(),
        image[-margin:,  -margin: ].ravel(),
    ])
    _, median, _ = sigma_clipped_stats(corners, sigma=3.0)
    print(f"  [Background] Estimated background = {median:.4f} counts")
    return image - median


def process_hst_image(fits_path, out_dir, galaxy_name, approach,
                       hst_pixel_scale=HST_PIXEL_SCALE):
    print(f"\n{'='*60}")
    print(f"Processing: {galaxy_name}  [{approach}]")
    print(f"Input: {fits_path}")

    with fits.open(fits_path) as hdul:
        # try SCI extension first (drizzled HST), then primary
        if "SCI" in [h.name for h in hdul]:
            image = hdul["SCI"].data.astype(np.float64)
            header = hdul["SCI"].header.copy()
            print(f"  Loaded SCI extension: {image.shape}")
        else:
            image = hdul[0].data.astype(np.float64)
            header = hdul[0].header.copy()
            print(f"  Loaded primary HDU: {image.shape}")

    if image.ndim == 3:
        image = image[0]

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    image = subtract_background(image)
    image = np.clip(image, 0, None)

    os.makedirs(out_dir, exist_ok=True)

    if approach in ("A", "both"):
        print(f"\n--- Approach A: No PSF matching / resampling ---")
        _run_pipeline(image.copy(), hst_pixel_scale,
                      out_dir, galaxy_name, tag="A_nopsf")

    if approach in ("B", "both"):
        print(f"\n--- Approach B: PSF match + area-average resample ---")
        img_b = psf_match(image.copy(),
                          src_fwhm_as=HST_PSF_FWHM_AS,
                          tgt_fwhm_as=SIM_PSF_FWHM_AS,
                          pixel_scale=hst_pixel_scale)
        img_b = resample_area_average(img_b, hst_pixel_scale, SIM_PIXEL_SCALE)
        _run_pipeline(img_b, SIM_PIXEL_SCALE,
                      out_dir, galaxy_name, tag="B_psf_resampled")


def _run_pipeline(image, pixel_scale, out_dir, galaxy_name, tag):
    print(f"  Image shape: {image.shape}  pixel scale: {pixel_scale}\"/px")

    plot_path = os.path.join(out_dir, f"{galaxy_name}_{tag}_ellipse_diagnostic.png")
    model = fit_ellipse_model(image, plot_path=plot_path)

    residual = compute_residual(image, model)
    residual = np.nan_to_num(residual, nan=0.0)

    residual_norm = normalise(residual)

    out_fits = os.path.join(out_dir, f"{galaxy_name}_{tag}_residual.fits")
    header_out = fits.Header()
    header_out["GALAXY"]  = galaxy_name
    header_out["APPROACH"]= tag
    header_out["PIXSCALE"]= (pixel_scale, "arcsec/px")
    header_out["RESIDUAL"]= (True, "(image-model)/sqrt(model), normalised")
    fits.PrimaryHDU(data=residual_norm, header=header_out).writeto(
        out_fits, overwrite=True)
    print(f"  Saved residual FITS: {out_fits}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{galaxy_name} — {tag}", fontsize=14, fontweight="bold")

    axes[0].imshow(residual_norm, cmap="gray", origin="lower",
                   vmin=0, vmax=1)
    axes[0].set_title("Normalised residual [0,1]", fontsize=12)
    axes[0].tick_params(labelsize=10)

    finite = residual_norm[residual_norm > 0]
    axes[1].hist(finite, bins=50, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Pixel value", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Residual pixel distribution", fontsize=12)
    axes[1].grid(True, alpha=0.4, linestyle="--")

    plt.tight_layout()
    plot_out = os.path.join(out_dir, f"{galaxy_name}_{tag}_residual_diagnostic.png")
    plt.savefig(plot_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved residual diagnostic: {plot_out}")


def main():
    parser = argparse.ArgumentParser(
        description="HST preprocessing pipeline for CNN SBF inference"
    )
    parser.add_argument("--fits",     required=True,
                        help="Path to HST FITS file")
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory")
    parser.add_argument("--galaxy",   required=True,
                        help="Galaxy name (e.g. NGC4458)")
    parser.add_argument("--approach", default="both",
                        choices=["A", "B", "both"],
                        help="A=no PSF match, B=PSF match+resample, both=run both")
    parser.add_argument("--hst_pixel_scale", type=float, default=0.13,
                        help="HST image pixel scale in arcsec/px (default 0.13)")
    args = parser.parse_args()

    process_hst_image(
        fits_path=args.fits,
        out_dir=args.out_dir,
        galaxy_name=args.galaxy,
        approach=args.approach,
        hst_pixel_scale=args.hst_pixel_scale,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
