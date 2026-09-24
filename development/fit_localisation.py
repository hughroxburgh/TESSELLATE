"""
Refit the SNR -> localisation-error model on manually sorted flares, and plot it
against the empirical offsets (same style as the empirical_loc notebook).

The sample is every event sorted as Flare from both lists: found_flares (a Gaia
star inside the old error region) and non_flares (none inside it). Together they
give offsets from the host star without the old model's cut, which is what kept
the previous floor too small. Offsets are event minus star, in pixels
(-nearest_gaia_dx/dy from the pipeline's crossmatch).

For each axis and containment percentage p it fits
    r_p(snr) = sqrt((a * snr**-b)**2 + floor**2)
by quantile regression: r_p is chosen so that p% of flares have |offset| <= r_p
at each SNR, using every event rather than binned percentiles. Events far out
only count as "outside", so a few wrong-host matches can't drag the curves.
The exponent b is shared by all percentages of an axis, and a and floor grow
with the percentage, so the curves never cross, even when extrapolated.

Written to OUT_DIR:
  fit_new_model.png        empirical bands with the refitted model
  fit_old_model.png        the same bands with the current package model
  coverage.png             fraction of flares inside each radius, per SNR bin
  report.txt               sample, bias, coverage (per axis and joint), parameters
  snr_localisation_model.pkl   the refit, in the format
                           tessellate.localisation.get_snr_to_localisation_func
                           reads (the package's model is left alone)

Edit the CONFIG block, then:  python fit_localisation.py
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.localisation import _bound_model, get_snr_to_localisation_func
from tessellate.ml_classifier import KEY_COLS, load_manual_labels

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
EVENT_CSVS = [f'{HERE}/found_flares.csv', f'{HERE}/non_flares.csv']              # the lists given to the sorter
SORT_DIRS = [f'{HERE}/S55/sort_found_flares', f'{HERE}/S55/sort_non_flares']     # their manual_sort outputs
LABEL = 'Flare'
CAMERAS = [1, 2, 3]         # camera 4 of found_flares wasn't fully sorted; including it would bias the sample

PERCENTAGES = [50, 60, 68, 75, 80, 85, 90, 95]   # levels fitted (the loader interpolates between them)
PLOT_PERCENTAGES = [50, 68, 90, 95]
N_BINS = 12                 # SNR bins (equal counts) for the empirical bands and coverage

OUT_DIR = f'{HERE}/localisation_fit'
# ----------------

COLORS = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown']


def load_sample():
    events = pd.concat([pd.read_csv(f, low_memory=False).assign(list=os.path.splitext(os.path.basename(f))[0])
                        for f in EVENT_CSVS], ignore_index=True)
    labels = pd.concat([load_manual_labels(d) for d in SORT_DIRS], ignore_index=True)
    df = events.merge(labels.loc[labels.label == LABEL, KEY_COLS], on=KEY_COLS)
    df = df[df.camera.isin(CAMERAS)].copy()
    df['dx'] = -pd.to_numeric(df['nearest_gaia_dx'], errors='coerce')
    df['dy'] = -pd.to_numeric(df['nearest_gaia_dy'], errors='coerce')
    return df.dropna(subset=['dx', 'dy', 'snr_psf']).reset_index(drop=True)


# ----------------------------- Fit ----------------------------- #

def _pinball(r, m, tau):
    d = r - m
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def _fit_levels(snr, r, b):
    """For a fixed exponent b: (total loss, {p: (a, b, floor)}), fitting the percentages in increasing order with
    a and floor each at least the previous level's -- so a higher percentage's radius is never smaller."""
    lo = snr <= np.quantile(snr, 1 / 6)
    hi = snr >= np.quantile(snr, 2 / 3)
    params, prev, total = {}, (0.0, 0.0), 0.0
    for p in PERCENTAGES:
        tau = p / 100

        def loss(u):
            return _pinball(r, _bound_model(snr, prev[0] + np.exp(u[0]), b, prev[1] + np.exp(u[1])), tau)

        f0 = max(np.quantile(r[hi], tau) - prev[1], 1e-3)
        a0 = max(np.sqrt(max(np.quantile(r[lo], tau) ** 2 - (prev[1] + f0) ** 2, 1e-6))
                 * np.median(snr[lo]) ** b - prev[0], 1e-3)
        best = min((minimize(loss, np.log([a0 * k, f0]), method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000}) for k in (0.3, 1, 3)),
                   key=lambda res: res.fun)
        a, f = prev[0] + np.exp(best.x[0]), prev[1] + np.exp(best.x[1])
        params[p], prev, total = (a, b, f), (a, f), total + best.fun
    return total, params


def fit_axis(snr, r):
    """{p: (a, b, floor)}: one exponent b shared by all percentages (the best total quantile loss), a and floor
    per percentage. Freeing b per percentage fits no better, and lets sparse high-SNR data cross the curves."""
    b = minimize_scalar(lambda b: _fit_levels(snr, r, b)[0], bounds=(0.2, 2.5), method='bounded',
                        options={'xatol': 1e-3}).x
    return _fit_levels(snr, r, b)[1]


def fit_model(df):
    model = {'percentages': np.array(PERCENTAGES), 'params': {}}
    for axis, col in [('x', 'dx'), ('y', 'dy')]:
        params = fit_axis(df.snr_psf.to_numpy(), np.abs(df[col].to_numpy()))
        model['params'][axis] = {p: [float(v) for v in params[p]] for p in PERCENTAGES}
    model['snr_range'] = (float(df.snr_psf.min()), float(df.snr_psf.max()))
    model['calibration_source'] = (f'sorted {LABEL} events, cameras {CAMERAS}, from '
                                   + ' + '.join(os.path.basename(f) for f in EVENT_CSVS)
                                   + ' (quantile regression, fit_localisation.py)')
    model['high_snr_cut'] = None
    return model


def crossing_check(model, snr_range):
    """Lines saying where a lower percentage's radius exceeds a higher one's."""
    grid = np.logspace(np.log10(snr_range[0]), np.log10(snr_range[1]), 400)
    lines = []
    for axis in ('x', 'y'):
        curves = np.array([_bound_model(grid, *model['params'][axis][p]) for p in PERCENTAGES])
        for i in range(len(PERCENTAGES) - 1):
            bad = curves[i] > curves[i + 1]
            if bad.any():
                lines.append(f'  {axis}: {PERCENTAGES[i]}% radius exceeds {PERCENTAGES[i + 1]}% '
                             f'for SNR {grid[bad].min():.1f}-{grid[bad].max():.1f}')
    return lines or ['  none: radii increase with percentage over the whole SNR range']


# ----------------------------- Diagnostics ----------------------------- #

def _snr_bins(df):
    return pd.qcut(df.snr_psf, N_BINS, duplicates='drop')


def binned_bands(df, col):
    rows = []
    for _, g in df.groupby(_snr_bins(df), observed=True):
        row = {'snr': g.snr_psf.median(), 'n': len(g), 'mean': g[col].mean()}
        for p in PLOT_PERCENTAGES:
            row[f'lo{p}'], row[f'hi{p}'] = np.percentile(g[col], [50 - p / 2, 50 + p / 2])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('snr')


def radii(func, snr, p):
    return np.atleast_1d(func(snr, p, 'x')), np.atleast_1d(func(snr, p, 'y'))


def coverage(df, func, p):
    """Fraction of flares inside the p% radius: x, y, and jointly inside the ellipse with those semi-axes."""
    rx, ry = radii(func, df.snr_psf.to_numpy(), p)
    ax, ay = np.abs(df.dx.to_numpy()), np.abs(df.dy.to_numpy())
    joint = np.hypot(ax / rx, ay / ry) <= 1
    return np.mean(ax <= rx), np.mean(ay <= ry), np.mean(joint)


def ellipse_scale(df, func, p=95, target=0.95):
    """Factor to scale the p% per-axis ellipse by so that `target` of flares fall inside it."""
    rx, ry = radii(func, df.snr_psf.to_numpy(), p)
    return np.quantile(np.hypot(df.dx.to_numpy() / rx, df.dy.to_numpy() / ry), target)


# ----------------------------- Plots ----------------------------- #

def plot_bands(df, func, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, axis in zip(axes, ['dx', 'dy'], ['x', 'y']):
        b = binned_bands(df, col)
        grid = np.logspace(np.log10(b.snr.min()), np.log10(b.snr.max()), 200)
        ax.plot(b.snr, b['mean'], color='k', lw=2, label='mean')
        for p, c in zip(PLOT_PERCENTAGES, COLORS):
            ax.fill_between(b.snr, b[f'lo{p}'], b[f'hi{p}'], color=c, alpha=0.12, label=f'{p}% empirical')
            r = func(grid, p, axis)
            ax.plot(grid, r, '--', color=c, lw=1.5, label=f'{p}% model')
            ax.plot(grid, -r, '--', color=c, lw=1.5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('SNR')
        ax.set_ylabel('offset (pixels)')
        ax.set_title(col)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(df, funcs, path):
    """Coverage per SNR bin for each model (solid = first, dotted = second)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    groups = list(df.groupby(_snr_bins(df), observed=True))
    snr = np.array([g.snr_psf.median() for _, g in groups])
    for (name, func), ls in zip(funcs.items(), ['-', ':']):
        for p, c in zip(PLOT_PERCENTAGES, COLORS):
            cov = np.array([coverage(g, func, p) for _, g in groups])
            for k in range(2):
                axes[k].plot(snr, cov[:, k], ls, color=c, marker='o', ms=3, label=f'{p}% {name}')
            if p == max(PLOT_PERCENTAGES):
                axes[2].plot(snr, cov[:, 2], ls, color=c, marker='o', ms=3, label=f'{p}% ellipse, {name}')
    for p, c in zip(PLOT_PERCENTAGES, COLORS):
        for k in range(2):
            axes[k].axhline(p / 100, color=c, lw=0.8, alpha=0.5)
    axes[2].axhline(max(PLOT_PERCENTAGES) / 100, color='gray', lw=0.8)
    for ax, title in zip(axes, ['|dx| <= r_x', '|dy| <= r_y', 'inside the ellipse (r_x, r_y)']):
        ax.set_xscale('log')
        ax.set_xlabel('SNR')
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel('fraction of flares inside')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------- Report ----------------------------- #

def report(df, model, new, old):
    lines = [f'Sample: {len(df)} events sorted as {LABEL}, cameras {CAMERAS}',
             '  ' + df.groupby(['list', 'psf_stacked']).size().rename('n').to_string().replace('\n', '\n  '),
             f'  snr_psf: min {df.snr_psf.min():.1f}, median {df.snr_psf.median():.1f}, max {df.snr_psf.max():.1f}', '']

    r = np.hypot(df.dx, df.dy)
    lines += ['Offset from the star (event - star, pixels):',
              f'  mean   dx {df.dx.mean():+.4f}  dy {df.dy.mean():+.4f}',
              f'  median dx {df.dx.median():+.4f}  dy {df.dy.median():+.4f}',
              f'  radial offset > 0.3 px: {np.sum(r > 0.3)}   > 0.5 px: {np.sum(r > 0.5)}   > 1 px: {np.sum(r > 1)}', '']

    # the pipeline's own match: star inside the ellipse with the event's stored 95% semi-axes
    stored = np.hypot(df.dx / df.xcentroid_err, df.dy / df.ycentroid_err) <= 1
    lines += ['Check: star inside the ellipse of the stored x/ycentroid_err (old 95% radii), by list:',
              '  ' + stored.groupby(df.list).mean().round(3).to_string().replace('\n', '\n  '), '']

    lines += ['Coverage (fraction of flares inside the radius): x / y / joint ellipse',
              f'  {"":6s} {"old model":>24s}   {"new model":>24s}']
    for p in PLOT_PERCENTAGES:
        o, n = coverage(df, old, p), coverage(df, new, p)
        lines.append(f'  {p:3d}%   ' + ' / '.join(f'{v:.3f}' for v in o) + '      ' + ' / '.join(f'{v:.3f}' for v in n))
    lines += ['', 'Coverage of the new 95% radii by psf_stacked: x / y / joint ellipse']
    for st, g in df.groupby('psf_stacked'):
        lines.append(f'  psf_stacked={st}  n={len(g):5d}  ' + ' / '.join(f'{v:.3f}' for v in coverage(g, new, 95)))
    lines += ['', 'Scale factor for the 95% per-axis ellipse to hold 95% of flares jointly:',
              f'  old model {ellipse_scale(df, old):.2f}   new model {ellipse_scale(df, new):.2f}',
              '  (for Gaussian errors this is 2.45 / 1.96 = 1.25)', '']

    lines += ['Crossing check (new model):'] + crossing_check(model, model['snr_range']) + ['']
    lines += ['New model parameters, r = sqrt((a * snr^-b)^2 + floor^2):']
    for axis in ('x', 'y'):
        for p in PERCENTAGES:
            a, b, f = model['params'][axis][p]
            lines.append(f'  {axis} {p:3d}%   a {a:9.4f}   b {b:6.3f}   floor {f:.4f}')
    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_sample()
    print(f'{len(df)} flares ({df.list.value_counts().to_dict()})')

    model = fit_model(df)
    model_path = f'{OUT_DIR}/snr_localisation_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    new = get_snr_to_localisation_func(model_path)    # evaluated exactly as the pipeline would
    old = get_snr_to_localisation_func()

    plot_bands(df, new, f'Refitted model: {len(df)} sorted flares, on- and off-star, cameras {CAMERAS}',
               f'{OUT_DIR}/fit_new_model.png')
    plot_bands(df, old, 'Current package model on the same flares', f'{OUT_DIR}/fit_old_model.png')
    plot_coverage(df, {'new': new, 'old': old}, f'{OUT_DIR}/coverage.png')

    text = report(df, model, new, old)
    with open(f'{OUT_DIR}/report.txt', 'w') as f:
        f.write(text)
    print('\n' + text + f'\nWritten to {OUT_DIR}')


if __name__ == '__main__':
    main()
