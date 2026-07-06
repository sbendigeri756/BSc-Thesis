import os
import itertools
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import gaussian_filter

import artpop

plt.style.use(artpop.mpl_style)

distances = np.logspace(np.log10(10), np.log10(150), 20).round(1).tolist()
metallicities = [-0.5, 0]
indices = [0.8, 1, 2]
num_stars = [1e8]
ages = [6, 10]
ellipticities = [0.0, 0.6]
pos_angles = [0, 60]
radii = [5, 10]

OUTPUT_DIR = os.environ.get(
    'ARTPOP_OUTDIR',
    '/scratch/s5511178/ResultsTrainingSet02PixScale1e8/'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_galaxy(params):
    dist, met, ind, num, age, ell, angle, rad = params

    num_str = f"{num:.0e}"
    label = f"spiral_d{dist}_feh{met}_n{ind}_s{num_str}_age{age}_e{ell}_pa{angle}_reff{rad}"

    raw_path = os.path.join(OUTPUT_DIR, f"{label}_raw.png")
    psf_path = os.path.join(OUTPUT_DIR, f"{label}_psf.png")
    fits_path = os.path.join(OUTPUT_DIR, f"{label}.fits")

    if all(os.path.exists(p) for p in [raw_path, psf_path, fits_path]):
        print(f"[SKIP] {label}", flush=True)
        return label, True

    print(f"[START] {label}", flush=True)

    seed = hash((dist, met, ind, num, age, ell, angle, rad)) % (2**31)
    rng = np.random.RandomState(seed)

    try:
        disk_spiral = artpop.MISTSersicSSP(
            log_age=age,
            feh=met,
            r_eff=rad * u.kpc,
            n=ind,
            theta=angle * u.deg,
            ellip=ell,
            num_stars=num,
            phot_system='UKIDSS',
            distance=dist * u.Mpc,
            xy_dim=501,
            pixel_scale=0.2,
            random_state=rng,
        )

        imager = artpop.IdealImager()
        obs = imager.observe(disk_spiral, 'UKIDSS_H')
        img = np.clip(obs.image, 0, None)

        log_img = np.log10(img + 1e-6)
        vmin, vmax = np.percentile(log_img, [30, 99.9])
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.imshow(log_img, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.savefig(raw_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

        img_psf = gaussian_filter(img, sigma=1.0)
        log_img_psf = np.log10(img_psf + 1e-6)
        vmin, vmax = np.percentile(log_img_psf, [30, 99.9])
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.imshow(log_img_psf, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax.axis('off')
        fig.savefig(psf_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

        hdu = fits.PrimaryHDU(data=img_psf)
        hdu.header['DISTANCE'] = (dist, 'Mpc')
        hdu.header['FEH'] = (met, '[Fe/H]')
        hdu.header['SERSIC'] = (ind, 'Sersic index')
        hdu.header['NSTARS'] = (num, 'total stars')
        hdu.header['LOGAGE'] = (age, 'log age in years')
        hdu.header['ELLIP'] = (ell, 'ellipticity 1-b/a')
        hdu.header['PA'] = (angle, 'position angle deg')
        hdu.header['REFF'] = (rad, 'effective radius kpc')
        hdu.writeto(fits_path, overwrite=True)

        print(f"[DONE] {label}", flush=True)
        return label, True

    except Exception as e:
        print(f"[ERROR] {label}: {e}", flush=True)
        return label, False

if __name__ == '__main__':
    all_params = list(itertools.product(
        distances, metallicities, indices, num_stars,
        ages, ellipticities, pos_angles, radii
    ))
    print(f"Total galaxies: {len(all_params)}", flush=True)

    n_workers = int(os.environ.get('NSLOTS', 4))
    print(f"Using {n_workers} worker processes", flush=True)

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(simulate_galaxy, all_params)

    n_ok = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_ok
    print(f"\nFinished: {n_ok} OK, {n_fail} failed", flush=True)
