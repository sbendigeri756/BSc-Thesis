#!/usr/bin/env python3

import os
import re
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits

from ml_alg_res_trial_7_0 import SBFEfficientNetB0, normalise

TITLE_FS = 16
LABEL_FS = 14
TICK_FS  = 12
ANNOT_FS = 9

LIT_DISTANCES = {
    "NGC1272": 71.0,
    "NGC3311": 41.2,
    "NGC3842": 87.5,
}

GALAXY_COLOURS = {
    "NGC1272": "steelblue",
    "NGC3311": "coral",
    "NGC3842": "mediumpurple",
    "NGC4458": "seagreen",
    "IC3586":  "goldenrod",
}
DEFAULT_COLOUR = "grey"

MARKER_CROP   = "o"  
MARKER_NOCROP = "s" 


def parse_galaxy_from_fname(fname):
    """
    Recognise known galaxy names anywhere in the filename (case-insensitive).
    Returns the canonical name (e.g. 'NGC1272') or None.
    """
    for gal in LIT_DISTANCES:
        if gal.lower() in fname.lower():
            return gal
    # fallback: read GALAXY header keyword (done later)
    return None


def parse_is_cropped(fname):
    """Return True if 'crop' appears in the filename."""
    return "crop" in fname.lower()


def load_ensemble(ckpt_dir, max_folds, dropout, device):
    models = []
    for fold in range(1, max_folds + 1):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] fold {fold} not found")
            continue
        model = SBFEfficientNetB0(pretrained=False, dropout=dropout)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        print(f"  Loaded fold {fold}  epoch={ckpt.get('epoch','?')}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
        models.append(model)
    print(f"  Ensemble: {len(models)} fold models")
    return models


@torch.no_grad()
def predict_single(models, image_np, device):
    img    = normalise(image_np.astype(np.float32))
    tensor = (torch.from_numpy(img)
              .unsqueeze(0).repeat(3, 1, 1)
              .unsqueeze(0).to(device))
    preds    = [m(tensor).item() for m in models]
    mean_log = float(np.mean(preds))
    std_log  = float(np.std(preds))
    mean_mpc = 10 ** mean_log
    std_mpc  = max(10 ** (mean_log + std_log) - mean_mpc, 0.0)
    return mean_mpc, std_mpc, preds

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device
                          if (torch.cuda.is_available() or
                              args.device == "cpu") else "cpu")
    print(f"Device: {device}")

    print("\nLoading ensemble checkpoints...")
    models = load_ensemble(args.ckpt_dir, args.max_folds, args.dropout, device)
    if not models:
        raise RuntimeError("No checkpoints loaded.")

    all_fits = sorted(glob.glob(os.path.join(args.fits_dir,
                                             "**", "*residual*.fits"),
                                recursive=True))
    fits_files = [f for f in all_fits if "A_nopsf" in os.path.basename(f)]
    if not fits_files:
        fits_files = all_fits   # fallback if naming differs
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {args.fits_dir}")

    print(f"\nFound {len(fits_files)} approach-A FITS files:")
    for f in fits_files:
        print(f"  {os.path.basename(f)}")

    print("\nRunning inference...")
    results = []
    for path in fits_files:
        fname = os.path.basename(path)
        try:
            with fits.open(path) as hdul:
                img = hdul[0].data.astype(np.float32)
                hdr = hdul[0].header
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            if img.ndim == 3:
                img = img[0]

            pred, std, fold_preds = predict_single(models, img, device)

            galaxy   = (parse_galaxy_from_fname(fname)
                        or hdr.get("GALAXY", None)
                        or fname)
            is_crop  = parse_is_cropped(fname)
            lit_dist = LIT_DISTANCES.get(galaxy, None)

            print(f"  {fname}")
            print(f"    galaxy={galaxy}  cropped={is_crop}  "
                  f"img={img.shape[0]}x{img.shape[1]}px")
            print(f"    pred={pred:.1f} ± {std:.1f} Mpc  "
                  f"lit={lit_dist} Mpc")

            results.append({
                "path":     path,
                "fname":    fname,
                "galaxy":   galaxy,
                "is_crop":  is_crop,
                "pred_mpc": pred,
                "std_mpc":  std,
                "lit_mpc":  lit_dist,
                "shape":    img.shape,
            })

        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    if not results:
        raise RuntimeError("No images successfully processed.")

    csv_path = os.path.join(args.out_dir, "hst_predictions.csv")
    with open(csv_path, "w") as f:
        f.write("filename,galaxy,cropped,pred_mpc,std_mpc,"
                "lit_mpc,error_mpc,frac_error_pct\n")
        for r in results:
            err  = r["pred_mpc"] - r["lit_mpc"] if r["lit_mpc"] else ""
            frac = (abs(r["pred_mpc"] - r["lit_mpc"]) / r["lit_mpc"] * 100
                    if r["lit_mpc"] else "")
            err_str  = f"{err:.2f}"  if isinstance(err,  float) else ""
            frac_str = f"{frac:.1f}" if isinstance(frac, float) else ""
            f.write(f"{r['fname']},{r['galaxy']},{r['is_crop']},"
                    f"{r['pred_mpc']:.2f},{r['std_mpc']:.2f},"
                    f"{r['lit_mpc'] or ''},"
                    f"{err_str},{frac_str}\n")
    print(f"\nSaved CSV -> {csv_path}")

    n = len(results)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.8), 7))
    fig.suptitle("CNN predictions on real HST galaxies",
                 fontsize=TITLE_FS + 2, fontweight="bold")

    x   = np.arange(n)
    pred_v   = np.array([r["pred_mpc"] for r in results])
    std_v    = np.array([r["std_mpc"]  for r in results])
    bar_cols = [GALAXY_COLOURS.get(r["galaxy"], DEFAULT_COLOUR)
                for r in results]
    xlabels  = [r["fname"]
                 .replace("_A_nopsf_residual.fits", "")
                 .replace("_residual.fits", "")
                 .replace(".fits", "")
                for r in results]

    bars = ax.bar(x, pred_v, color=bar_cols, alpha=0.8,
                  edgecolor="none", width=0.6)
    ax.errorbar(x, pred_v, yerr=std_v, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=5)

    for bar, pred, std in zip(bars, pred_v, std_v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1.5,
                f"{pred:.1f}",
                ha="center", va="bottom",
                fontsize=ANNOT_FS, fontweight="bold", color="black")

    drawn_gals = set()
    for i, r in enumerate(results):
        gal = r["galaxy"]
        lit = r["lit_mpc"]
        if lit is None or gal in drawn_gals:
            continue
        col = GALAXY_COLOURS.get(gal, DEFAULT_COLOUR)
        ax.axhline(lit, color=col, lw=1.8, ls="--", alpha=0.9)
        # annotate the literature value at the right edge in matching colour
        ax.text(n - 0.5, lit + 0.8, f"{gal}: {lit} Mpc",
                color=col, fontsize=ANNOT_FS, va="bottom", ha="right",
                fontweight="bold")
        drawn_gals.add(gal)

    legend_handles = []
    for gal in sorted(set(r["galaxy"] for r in results)):
        col = GALAXY_COLOURS.get(gal, DEFAULT_COLOUR)
        legend_handles.append(
            Line2D([0], [0], color=col, lw=10, alpha=0.8, label=gal)
        )
    legend_handles.append(
        Line2D([0], [0], color="grey", lw=1.8, ls="--",
               label="Literature value (dotted)")
    )
    ax.legend(handles=legend_handles, fontsize=TICK_FS,
              loc="upper left", framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=ANNOT_FS)
    ax.set_ylabel("Predicted distance (Mpc)", fontsize=LABEL_FS)
    ax.set_title("Predicted distances (approach A: no PSF match)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "hst_bar.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart -> {out1}")

    labelled = [r for r in results if r["lit_mpc"] is not None]
    if len(labelled) >= 2:
        fig, ax = plt.subplots(figsize=(8, 7))

        all_true = [r["lit_mpc"]  for r in labelled]
        all_pred = [r["pred_mpc"] for r in labelled]
        lim_lo   = min(all_true + all_pred) * 0.7
        lim_hi   = max(all_true + all_pred) * 1.2

        #1:1 line
        d_range = np.linspace(lim_lo, lim_hi, 200)
        ax.plot(d_range, d_range, "k--", lw=1.5, label="1:1", zorder=1)

        #±20% shaded band 
        ax.fill_between(d_range, 0.8*d_range, 1.2*d_range,
                        alpha=0.08, color="grey", label="±20%")

        legend_handles = [
            Line2D([0], [0], color="k",    lw=1.5, ls="--", label="1:1"),
            Line2D([0], [0], color="grey", lw=0,
                   marker="s", markersize=8, alpha=0.3, label="±20%"),
        ]
        seen_gals  = set()
        seen_crops = set()

        for r in labelled:
            col    = GALAXY_COLOURS.get(r["galaxy"], DEFAULT_COLOUR)
            marker = MARKER_CROP if r["is_crop"] else MARKER_NOCROP
            ax.scatter(r["lit_mpc"], r["pred_mpc"],
                       s=160, color=col, marker=marker,
                       edgecolor="none", zorder=3, alpha=0.9)
            ax.errorbar(r["lit_mpc"], r["pred_mpc"],
                        yerr=r["std_mpc"],
                        fmt="none", ecolor="grey",
                        elinewidth=1.2, capsize=4, zorder=2)

            if r["galaxy"] not in seen_gals:
                legend_handles.append(
                    Line2D([0], [0], color=col, lw=0,
                           marker="o", markersize=10,
                           label=r["galaxy"])
                )
                seen_gals.add(r["galaxy"])

        legend_handles += [
            Line2D([0], [0], color="grey", lw=0,
                   marker=MARKER_CROP,   markersize=9,
                   label="Cropped image"),
            Line2D([0], [0], color="grey", lw=0,
                   marker=MARKER_NOCROP, markersize=9,
                   label="No crop"),
        ]

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("Literature distance (Mpc)", fontsize=LABEL_FS)
        ax.set_ylabel("CNN predicted distance (Mpc)", fontsize=LABEL_FS)
        ax.set_title("Predicted vs literature distance",
                     fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, alpha=0.4, linestyle="--")

        ax.legend(handles=legend_handles,
                  fontsize=TICK_FS,
                  loc="upper left",
                  bbox_to_anchor=(1.02, 1),
                  borderaxespad=0,
                  framealpha=0.9)

        plt.tight_layout()
        out2 = os.path.join(args.out_dir, "hst_1to1.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved 1:1 plot -> {out2}")
    else:
        print("Not enough labelled galaxies for 1:1 plot — skipping")

    print("\n" + "="*55)
    print("Summary")
    print("="*55)
    print(f"  {'Galaxy':<12} {'Crop':<6} {'Pred':>8}  {'±':>6}  {'Lit':>8}  {'Err':>8}")
    print("-"*55)
    for r in results:
        lit_str = f"{r['lit_mpc']:.1f}" if r["lit_mpc"] else "?"
        err_str = (f"{r['pred_mpc']-r['lit_mpc']:+.1f}"
                   if r["lit_mpc"] else "?")
        print(f"  {r['galaxy']:<12} {'yes' if r['is_crop'] else 'no':<6} "
              f"{r['pred_mpc']:>7.1f}  "
              f"{r['std_mpc']:>5.1f}  "
              f"{lit_str:>7}  {err_str:>7} Mpc")

    print(f"\nNote: NGC4458/IC3586 are below the training range (>40 Mpc).")
    print(f"      NGC1272/NGC3311/NGC3842 are within training range.")
    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN inference on real HST galaxy residuals"
    )
    parser.add_argument("--fits_dir",  required=True)
    parser.add_argument("--ckpt_dir",  required=True)
    parser.add_argument("--out_dir",   default="./hst_predictions")
    parser.add_argument("--max_folds", type=int,   default=1)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--device",    type=str,   default="cpu")
    args = parser.parse_args()
    main(args)
