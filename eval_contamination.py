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


TITLE_FS = 20
LABEL_FS = 18
TICK_FS  = 18
ANNOT_FS = 16

COLOUR_MAP = {1: "tab:blue", 2: "tab:orange", 3: "tab:red"}
MARKER_MAP = {5: "o", 10: "s"}

def parse_dist_from_path(path):
    b = os.path.basename(str(path))
    for pat in [r"_dist([\d.]+)_", r"_d([\d.]+)_", r"([\d.]+)Mpc"]:
        m = re.search(pat, b, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def parse_reff_from_path(path):
    b = os.path.basename(str(path))
    m = re.search(r"_r(?:eff)?(\d+(?:\.\d+)?)_", b)
    return float(m.group(1)) if m else np.nan

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
    tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    preds  = [m(tensor).item() for m in models]
    mean_log = float(np.mean(preds))
    std_log  = float(np.std(preds))
    mean_mpc = 10 ** mean_log
    std_mpc  = max(10 ** (mean_log + std_log) - mean_mpc, 0.0)
    return mean_mpc, std_mpc

def compute_metrics(true_mpc, pred_mpc):
    if len(true_mpc) < 2:
        return None
    frac = np.abs(pred_mpc - true_mpc) / true_mpc
    bias = np.mean(pred_mpc - true_mpc)
    return {
        "N":             len(true_mpc),
        "MAE_Mpc":       float(np.mean(np.abs(pred_mpc - true_mpc))),
        "MAE_pct":       float(frac.mean() * 100),
        "RMSE_Mpc":      float(np.sqrt(np.mean((pred_mpc - true_mpc)**2))),
        "MedianErr_Mpc": float(np.median(np.abs(pred_mpc - true_mpc))),
        "Bias_Mpc":      float(bias),
        "R2":            float(r2_score(np.log10(true_mpc), np.log10(pred_mpc))),
        "within_5pct":   float((frac <= 0.05).mean() * 100),
        "within_10pct":  float((frac <= 0.10).mean() * 100),
        "within_20pct":  float((frac <= 0.20).mean() * 100),
    }


def print_metrics(label, m):
    if m is None:
        print(f"  {label}: insufficient data")
        return
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"{label}")
    print("-" * 52)
    print(f"  N                  = {m['N']}")
    print(f"  MAE (Mpc)          = {m['MAE_Mpc']:.2f}")
    print(f"  MAE (%)            = {m['MAE_pct']:.1f}%")
    print(f"  RMSE (Mpc)         = {m['RMSE_Mpc']:.2f}")
    print(f"  Median residual    = {m['MedianErr_Mpc']:.2f} Mpc")
    print(f"  Mean residual/bias = {m['Bias_Mpc']:.2f} Mpc")
    print(f"  R²                 = {m['R2']:.4f}")
    print(f"  Within  5%         = {m['within_5pct']:.1f}%")
    print(f"  Within 10%         = {m['within_10pct']:.1f}%")
    print(f"  Within 20%         = {m['within_20pct']:.1f}%")


def plot_1to1(true_mpc, pred_mpc, n_sources, reffs_arr, out_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_reffs = sorted(set(r for r in reffs_arr if not np.isnan(r)))
    for n in sorted(COLOUR_MAP):
        for rv in unique_reffs:
            mask = (n_sources == n) & (reffs_arr == rv)
            if mask.sum() == 0:
                continue
            marker = MARKER_MAP.get(rv, "D")
            ax.scatter(true_mpc[mask], pred_mpc[mask],
                   color=COLOUR_MAP[n], s=100,
                   marker=marker,
                   edgecolor="none",
                   label=f"{n} src, r_eff={rv:.0f} kpc")

    lims = [min(true_mpc.min(), pred_mpc.min()) * 0.9,
            max(true_mpc.max(), pred_mpc.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1.5, label="1:1")
    ax.set_xlabel("True distance (Mpc)", fontsize=LABEL_FS)
    ax.set_ylabel("Predicted distance (Mpc)", fontsize=LABEL_FS)
    ax.set_title("Predicted vs true - point source contamination",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
    fig.subplots_adjust(right=0.75)
    ax.grid(True, alpha=0.4, linestyle="--")

    frac = np.abs(pred_mpc - true_mpc) / true_mpc
    txt = (f"N={len(true_mpc)}\n"
           f"MAE%={frac.mean()*100:.1f}%\n"
           f"Within 5%: {(frac<=0.05).mean()*100:.1f}%\n"
           f"Within 10%: {(frac<=0.10).mean()*100:.1f}%")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=ANNOT_FS,
            va="top", bbox=dict(boxstyle="round", fc="white", ec="grey"))

    plt.tight_layout()
    out = os.path.join(out_dir, "pointsource_1to1.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def plot_residuals(true_mpc, pred_mpc, n_sources, reffs_arr, out_dir):
    residuals = pred_mpc - true_mpc
    lims = [true_mpc.min() * 0.9, true_mpc.max() * 1.1]
    d_range = np.linspace(lims[0], lims[1], 200)

    fig, ax = plt.subplots(figsize=(11, 5))
    unique_reffs = sorted(set(r for r in reffs_arr if not np.isnan(r)))
    for n in sorted(COLOUR_MAP):
        for rv in unique_reffs:
            mask = (n_sources == n) & (reffs_arr == rv)
            if mask.sum() == 0:
                continue
            marker = MARKER_MAP.get(rv, "D")
            ax.scatter(true_mpc[mask], residuals[mask],
                   color=COLOUR_MAP[n], s=100,
                   marker=marker,
                   edgecolor="none",
                   label=f"{n} src, r_eff={rv:.0f} kpc")

    ax.axhline(0, color="k", lw=1.5, ls="--")
    ax.fill_between(d_range, -0.05*d_range,  0.05*d_range,
                    alpha=0.12, color="green",  label="±5%")
    ax.fill_between(d_range, -0.10*d_range,  0.10*d_range,
                    alpha=0.08, color="orange", label="±10%")
    ax.set_xlabel("True distance (Mpc)", fontsize=LABEL_FS)
    ax.set_ylabel("Predicted - True (Mpc)", fontsize=LABEL_FS)
    ax.set_title("Residuals - point source contamination",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
    fig.subplots_adjust(right=0.75)
    ax.grid(True, alpha=0.4, linestyle="--")

    plt.tight_layout()
    out = os.path.join(out_dir, "pointsource_residuals.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    print("\nLoading ensemble...")
    models = load_ensemble(args.ckpt_dir, args.max_folds, args.dropout, device)
    if not models:
        raise RuntimeError("No checkpoints loaded.")

    print(f"\nLoading NPZ: {args.npz}")
    data   = np.load(args.npz, allow_pickle=True)
    images = data["images"]
    paths  = list(data["paths"])
    print(f"  {len(images)} images  |  keys: {list(data.files)}")

    if "log_distances" in data.files:
        true_mpcs = list(10 ** data["log_distances"].astype(np.float32))
    else:
        true_mpcs = [parse_dist_from_path(p) for p in paths]

    if "n_point_sources" in data.files:
        n_sources_all = list(data["n_point_sources"].astype(int))
        print(f"  n_point_sources from NPZ: {sorted(set(n_sources_all))}")
    else:
        print("  WARNING: n_point_sources not in NPZ — defaulting to 1 for all")
        n_sources_all = [1] * len(images)

    print(f"\nRunning inference on {len(images)} images...")
    pred_mpcs, std_mpcs = [], []
    for i in range(len(images)):
        img = images[i]
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if img.ndim == 3:
            img = img[0]
        pred, std = predict_single(models, img, device)
        pred_mpcs.append(pred)
        std_mpcs.append(std)
        if i % 10 == 0 or i == len(images) - 1:
            t = true_mpcs[i]
            t_str = f"{t:.1f}" if t is not None else "?"
            print(f"  [{i+1:4d}/{len(images)}]  true={t_str:>7} Mpc  "
                  f"pred={pred:7.1f} Mpc  n_src={n_sources_all[i]}")

    pred_mpcs    = np.array(pred_mpcs)
    std_mpcs     = np.array(std_mpcs)
    n_sources_arr = np.array(n_sources_all)

    labelled_mask = np.array([t is not None for t in true_mpcs])
    t_arr = np.array([t for t in true_mpcs if t is not None])
    p_arr = pred_mpcs[labelled_mask]
    n_arr = n_sources_arr[labelled_mask]
    paths_arr = np.array(paths)[labelled_mask]
    reffs_arr = np.array([parse_reff_from_path(p) for p in paths_arr])
    # overall metrics
    print_metrics("Overall", compute_metrics(t_arr, p_arr))

    for n in sorted(set(n_arr)):
        mask = n_arr == n
        label = f"{n} point source{'s' if n > 1 else ''}"
        print_metrics(label, compute_metrics(t_arr[mask], p_arr[mask]))

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("filename,true_mpc,pred_mpc,std_mpc,n_point_sources,"
                "error_mpc,frac_error_pct\n")
        for path, true, pred, std, n in zip(
                paths_arr, t_arr, p_arr, std_mpcs[labelled_mask], n_arr):
            err  = pred - true
            frac = abs(err) / true * 100
            f.write(f"{os.path.basename(str(path))},"
                    f"{true:.2f},{pred:.2f},{std:.2f},{n},"
                    f"{err:.2f},{frac:.1f}\n")
    print(f"\nSaved CSV -> {csv_path}")

    plot_1to1(t_arr, p_arr, n_arr, reffs_arr,  args.out_dir)
    plot_residuals(t_arr, p_arr, n_arr, reffs_arr, args.out_dir)

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",       required=True)
    parser.add_argument("--ckpt_dir",  required=True)
    parser.add_argument("--out_dir",   default="./contamination_results")
    parser.add_argument("--max_folds", type=int,   default=5)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--device",    type=str,   default="cpu")
    args = parser.parse_args()
    main(args)
