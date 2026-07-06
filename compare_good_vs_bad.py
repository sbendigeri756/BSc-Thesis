#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from ml_alg_res_trial_7_0 import SBFEfficientNetB0

#get info from filename
def parse_param(path, key):
    """Parse a named parameter from the filename."""
    patterns = {
        "reff":   r"_r(?:eff)?(\d+(?:\.\d+)?)_",
        "n":      r"_n(\d+(?:\.\d+)?)_",
        "nstars": r"_s([\d.e+]+)_",
        "age":    r"_age(\d+(?:\.\d+)?)_",
        "feh":    r"_feh([+-]?\d+(?:\.\d+)?)_",
        "ellip":  r"_e(\d+(?:\.\d+)?)_",
    }
    m = re.search(patterns[key], os.path.basename(str(path)))
    return float(m.group(1)) if m else np.nan


def parse_all_params(path):
    keys = ["reff", "n", "nstars", "age", "feh", "ellip"]
    return {k: parse_param(path, k) for k in keys}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x):
        return self.model(x)

    def heatmap(self):
        w = self.gradients.mean(dim=[0, 2, 3])
        act = self.activations.clone()
        for i in range(act.shape[1]):
            act[:, i] *= w[i]
        cam = act.mean(dim=1).squeeze()
        cam = torch.clamp(cam, min=0)
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

#we want to match the model exactly
def preprocess(image):
    lo, hi = np.percentile(image, [0.5, 99.5])
    img = np.clip(image, lo, hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)


def run_inference(model, images, log_distances, valid_indices, device):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(valid_indices):
            if i % 1000 == 0:
                print(f"  inference {i}/{len(valid_indices)}")
            t = preprocess(images[idx]).unsqueeze(0).to(device)
            preds.append(model(t).item())
            targets.append(log_distances[idx])
    return np.array(preds), np.array(targets)

#gradcam for one image
def get_gradcam(model, cam, image, device):
    H, W = image.shape
    model.zero_grad()
    t = preprocess(image).unsqueeze(0).to(device)
    t.requires_grad_(True)
    out = cam(t)
    out.backward()
    hm = cam.heatmap().unsqueeze(0).unsqueeze(0)
    hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)
    return hm.squeeze().cpu().numpy()


def plot_panel(ax_img, ax_cam, ax_txt, image, heatmap, params,
               true_mpc, pred_mpc, error_mpc, title_colour):
    lo, hi = np.percentile(image, [0.5, 99.5])
    img_disp = np.clip((image - lo) / max(hi - lo, 1e-8), 0, 1)
    ax_img.imshow(img_disp, cmap="gray", origin="lower")
    ax_img.axis("off")
    ax_img.set_title(
        f"True={true_mpc:.1f}  Pred={pred_mpc:.1f} Mpc\nErr={error_mpc:+.1f} Mpc",
        fontsize=18, color=title_colour
    )

    ax_cam.imshow(img_disp, cmap="gray", origin="lower")
    ax_cam.imshow(heatmap, cmap="jet", alpha=0.45, origin="lower")
    ax_cam.axis("off")
    ax_cam.set_title("Grad-CAM", fontsize=7)

    ax_txt.axis("off")
    txt = "\n".join([
        f"r_eff : {params['reff']:.0f} kpc",
        f"n     : {params['n']:.1f}",
        f"stars : {params['nstars']:.0e}",
        f"age   : {params['age']:.0f} Gyr",
        f"[Fe/H]: {params['feh']:.1f}",
        f"ellip : {params['ellip']:.1f}",
        f"size  : {image.shape[0]}×{image.shape[1]} px",
    ])
    ax_txt.text(0.05, 0.95, txt, transform=ax_txt.transAxes,
                fontsize=16, va="top", family="monospace",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

#compare good vs bad per stripe
def make_comparison_figure(stripe_name, good_cases, bad_cases,
                           images, paths, cam, model, device, out_dir):
    n_good = len(good_cases)
    n_bad  = len(bad_cases)
    n_cols = max(n_good, n_bad)

    fig = plt.figure(figsize=(n_cols * 3.5, 14))
    fig.suptitle(
        f"Good vs Bad predictions — {stripe_name}\n"
        f"Left: well-predicted  |  Right-ish: badly-predicted",
        fontsize=16, weight="bold"
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    for group_idx, (cases, group_label, col) in enumerate([
        (good_cases, "GOOD predictions", "darkgreen"),
        (bad_cases,  "BAD predictions",  "darkred"),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, max(len(cases), 1),
            subplot_spec=outer[group_idx], hspace=0.1, wspace=0.05
        )
        fig.text(0.01, 0.73 - group_idx * 0.47, group_label,
                 color=col, fontsize=18, weight="bold", va="top")

        for ci, case in enumerate(cases):
            idx   = case["valid_idx"]
            image = images[idx]
            hm    = get_gradcam(model, cam, image, device)
            params = parse_all_params(paths[idx])

            ax_img = fig.add_subplot(inner[0, ci])
            ax_cam = fig.add_subplot(inner[1, ci])
            ax_txt = fig.add_subplot(inner[2, ci])
            plot_panel(ax_img, ax_cam, ax_txt, image, hm, params,
                       case["true_mpc"], case["pred_mpc"], case["error_mpc"],
                       title_colour=col)

    out = os.path.join(out_dir, f"good_vs_bad_{stripe_name.replace(' ', '_')}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def make_param_summary(stripe_name, good_cases, bad_cases,
                       images, paths, out_dir):
    def collect(cases):
        reffs, ns, stars, ages = [], [], [], []
        for c in cases:
            p = parse_all_params(paths[c["valid_idx"]])
            reffs.append(p["reff"])
            ns.append(p["n"])
            stars.append(p["nstars"])
            ages.append(p["age"])
        return reffs, ns, stars, ages

    g_reff, g_n, g_stars, g_ages = collect(good_cases)
    b_reff, b_n, b_stars, b_ages = collect(bad_cases)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"Parameter distribution: good vs bad — {stripe_name}", fontsize=18)

    def bar_compare(ax, g_vals, b_vals, title):
        all_vals = sorted(set(g_vals + b_vals))
        x = np.arange(len(all_vals))
        w = 0.35
        g_counts = [g_vals.count(v) for v in all_vals]
        b_counts = [b_vals.count(v) for v in all_vals]
        ax.bar(x - w/2, g_counts, w, label="Good", color="green", alpha=0.7)
        ax.bar(x + w/2, b_counts, w, label="Bad",  color="red",   alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in all_vals], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Count")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    bar_compare(axes[0], g_reff,  b_reff,  "r_eff (kpc)")
    bar_compare(axes[1], g_n,     b_n,     "Sérsic n")
    bar_compare(axes[2], g_stars, b_stars, "Num stars")
    bar_compare(axes[3], g_ages,  b_ages,  "Stellar age (Gyr)")

    plt.tight_layout()
    out = os.path.join(out_dir, f"param_dist_{stripe_name.replace(' ', '_')}.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved → {out}")


#how many are well predicted in each stripe (histogram)
def reff_confusion_summary(stripe_name, stripe_pos, valid_indices,
                            true_mpc, preds_mpc, paths, out_dir,
                            good_pct=5.0):
    threshold = good_pct / 100.0   # fractional error threshold

    reffs_all  = []
    well_flags = []

    for p in stripe_pos:
        actual_idx = valid_indices[p]
        t = true_mpc[p]
        pr = preds_mpc[p]
        frac_err = abs(pr - t) / t
        reff = parse_param(paths[actual_idx], "reff")
        reffs_all.append(reff)
        well_flags.append(frac_err <= threshold)

    reffs_all  = np.array(reffs_all)
    well_flags = np.array(well_flags)

    unique_reffs = sorted(set(r for r in reffs_all if not np.isnan(r)))

    print(f"\n  r_eff confusion summary for {stripe_name}  "
          f"(good = fractional error ≤ {good_pct:.0f}%)")
    print(f"  {'r_eff':>8}  {'well':>6}  {'badly':>6}  {'total':>6}  "
          f"{'% well':>8}")
    print(f"  {'-'*44}")
    rows = []
    for reff in unique_reffs:
        mask  = reffs_all == reff
        n_well  = int(well_flags[mask].sum())
        n_badly = int((~well_flags[mask]).sum())
        n_total = n_well + n_badly
        pct_well = 100.0 * n_well / n_total if n_total > 0 else 0.0
        print(f"  {reff:>6.0f} kpc  {n_well:>6}  {n_badly:>6}  "
              f"{n_total:>6}  {pct_well:>7.1f}%")
        rows.append((reff, n_well, n_badly, pct_well))

    n_well_tot  = int(well_flags.sum())
    n_badly_tot = len(well_flags) - n_well_tot
    pct_tot = 100.0 * n_well_tot / len(well_flags) if len(well_flags) > 0 else 0.0
    print(f"  {'ALL':>8}  {n_well_tot:>6}  {n_badly_tot:>6}  "
          f"{len(well_flags):>6}  {pct_tot:>7.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"r_eff confusion summary — {stripe_name}\n"
        f"'Well predicted' = fractional error ≤ {good_pct:.0f}%",
        fontsize=16
    )

    ax = axes[0]
    x      = np.arange(len(rows))
    labels = [f"{r[0]:.0f} kpc" for r in rows]
    wells  = [r[1] for r in rows]
    badlys = [r[2] for r in rows]
    ax.bar(x, wells,  label=f"Well (≤{good_pct:.0f}%)",  color="green", alpha=0.75)
    ax.bar(x, badlys, bottom=wells, label=f"Badly (>{good_pct:.0f}%)", color="red", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Counts by r_eff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    # annotate totals
    for xi, (w, b) in enumerate(zip(wells, badlys)):
        ax.text(xi, w + b + 0.5, str(w + b),
                ha="center", va="bottom", fontsize=15, fontweight="bold")

    ax = axes[1]
    pct_wells = [r[3] for r in rows]
    colours   = ["green" if p >= 50 else "red" for p in pct_wells]
    bars = ax.bar(x, pct_wells, color=colours, alpha=0.75)
    ax.axhline(good_pct, color="orange", ls="--", lw=1.5,
               label=f"{good_pct:.0f}% threshold")
    ax.axhline(50, color="grey", ls=":", lw=1, label="50% line")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("% well predicted")
    ax.set_ylim(0, 105)
    ax.set_title("% well predicted by r_eff")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, pct in zip(bars, pct_wells):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir,
                       f"reff_confusion_{stripe_name.replace(' ', '_')}.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved → {out}")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading NPZ...")
    data          = np.load(args.npz, allow_pickle=True)
    images        = data["images"]
    log_distances = data["log_distances"].astype(np.float32)
    paths         = list(data["paths"])

    dists_mpc    = 10 ** log_distances
    valid_indices = np.where(dists_mpc > 40.0)[0]
    print(f"Valid samples (d>40 Mpc): {len(valid_indices)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SBFEfficientNetB0(pretrained=False, dropout=args.dropout)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)

    print("Running inference...")
    preds_log, targets_log = run_inference(
        model, images, log_distances, valid_indices, device
    )
    preds_mpc  = 10 ** preds_log
    true_mpc   = 10 ** targets_log
    errors_mpc = preds_mpc - true_mpc   # signed
    abs_errors = np.abs(errors_mpc)

    stripes = [
        ("stripe_72Mpc",  1.807, 1.907),   # ~64–81 Mpc
        ("stripe_145Mpc", 2.111, 2.211),   # ~129–162 Mpc
    ]

    for stripe_name, log_lo, log_hi in stripes:
        print(f"\n--- Processing {stripe_name} ---")
        stripe_mask = (targets_log >= log_lo) & (targets_log < log_hi)
        stripe_pos  = np.where(stripe_mask)[0]  # positions in the valid_indices array

        if len(stripe_pos) == 0:
            print(f"  No samples in this stripe — skipping.")
            continue

        stripe_errors = abs_errors[stripe_pos]
        stripe_true   = true_mpc[stripe_pos]
        stripe_pred   = preds_mpc[stripe_pos]

        #sort by absolute error
        sorted_order = np.argsort(stripe_errors)

        #best N and worst N
        n_each = args.n_each
        good_pos = sorted_order[:n_each]   # lowest errors
        bad_pos  = sorted_order[-n_each:]  # highest errors

        def make_cases(positions):
            cases = []
            for p in positions:
                orig_pos   = stripe_pos[p]
                actual_idx = valid_indices[orig_pos]
                cases.append({
                    "valid_idx": actual_idx,
                    "true_mpc":  float(stripe_true[p]),
                    "pred_mpc":  float(stripe_pred[p]),
                    "error_mpc": float(preds_mpc[orig_pos] - true_mpc[orig_pos]),
                })
            return cases

        good_cases = make_cases(good_pos)
        bad_cases  = make_cases(bad_pos)

        good_errs = [f"{c['error_mpc']:+.1f}" for c in good_cases]
        bad_errs  = [f"{c['error_mpc']:+.1f}" for c in bad_cases]
        print(f"  Good: errors = {good_errs}")
        print(f"  Bad:  errors = {bad_errs}")
        
        make_comparison_figure(stripe_name, good_cases, bad_cases,
                               images, paths, cam, model, device, args.out_dir)
        make_param_summary(stripe_name, good_cases, bad_cases,
                           images, paths, args.out_dir)

        # full-stripe reff confusion summary (uses ALL galaxies in stripe,
        # not just the top/bottom n_each)
        reff_confusion_summary(
            stripe_name, stripe_pos, valid_indices,
            true_mpc, preds_mpc, paths, args.out_dir,
            good_pct=args.good_pct,
        )

    cam.remove()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare good vs bad predictions at degenerate distance stripes"
    )
    parser.add_argument("--npz",        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir",    default="./good_vs_bad")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--n_each",     type=int,   default=6,
                        help="Number of good and bad cases to show per stripe")
    parser.add_argument("--good_pct",   type=float, default=5.0,
                        help="Fractional error threshold in %% for 'well predicted'")
    args = parser.parse_args()
    main(args)
