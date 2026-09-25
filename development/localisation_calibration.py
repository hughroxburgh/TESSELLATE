"""
Regenerate the localisation constants in tessellate/localisation.py:
    OPTICAL_AXIS_PX, RADIAL_SHIFT_COEF, LOCALISATION_A, LOCALISATION_B, LOCALISATION_FLOOR

1. Optical axis: each camera's focal-plane origin, in each CCD's 0-based FFI pixels (TESSELLATE's xccd/yccd), from
   tess-point's focal-plane geometry (pip package tess-point, module tess_stars2px).
2. Radial shift: PSF-fit positions of flare stars sit closer to the camera's optical axis than their Gaia star, by
   an amount that depends only on the distance r from the axis. The component toward the axis is fitted as a cubic
   in r through binned medians of CALIB_CSV (flare stars, variable == 0; dx, dy = raw PSF fit - Gaia star, px).
3. Injection core: recovered injected flares in INJ_CSV (dx, dy = recovered - injected), fitted with a Gaussian core
   plus a wide outlier component: sigma_inj(snr) = sqrt((A snr^-B)^2 + floor_inj^2). The same PRF injects and fits,
   so this is the fitting noise alone.
4. Real-sky floor: manually sorted flares (EVENT_CSVS + LABELS) with the radial shift removed, fitted with
   sigma(snr) = sqrt(sigma_inj(snr)^2 + sys^2) plus outliers. LOCALISATION_FLOOR = sqrt(floor_inj^2 + sys^2).
   A fit with A, B and the floor all free on the sorted flares is reported as a check.

The model: 2D circular offsets, so the radial offset follows a Rayleigh mixture
    p(r) = (1 - eps) Rayleigh(r; sigma(snr)) + eps Rayleigh(r; sig_out),  eps = expit(e0 + e1 log10(snr / SNR_REF))
truncated at the sample's maximum offset. sigma is the 1-sigma error per axis (centroid_err in the pipeline).

The inputs for the current constants are in development/localisation_calibration_data/, trimmed to the columns
used here; sort_labels_S55.csv is the record of the manual sort (from the sorted images' names).
Writes OUT_DIR/report.txt and plots, and prints the constants block to paste into tessellate/localisation.py.
Edit the CONFIG block, then:  python localisation_calibration.py
"""

import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
DATA = f'{HERE}/localisation_calibration_data'        # the inputs used for the constants in localisation.py
CALIB_CSV = f'{DATA}/all_events_S27-39.csv'           # flare stars (variable == 0) from a run without GLOBAL_PSF_Y_OFFSET
INJ_CSV = f'{DATA}/injections_S32_S35_S38.csv'        # SourceInjector.filter_transients() output, several cuts
EVENT_CSVS = [f'{DATA}/found_flares_S55.csv', f'{DATA}/non_flares_S55.csv']      # event lists given to the sorter
LABELS = f'{DATA}/sort_labels_S55.csv'                # labels as a csv (KEY + label), or a list of manual_sort folders
LABEL = 'Flare'
CAMERAS = [1, 2, 3, 4]
ASSUME_FLARE = {EVENT_CSVS[0]: [4]}    # unsorted events of this list in these cameras count as LABEL
                                       # (found_flares were 1512/1517 flares in cameras 1-3 of S55)
Y_OFFSET_IN_PSF = 0.0226               # y offset the sorted events' run subtracted from ycentroid_psf
                                       # (GLOBAL_PSF_Y_OFFSET for S55; 0 for runs of this version)

RADIAL_RANGE = (200, 3000)             # px from the axis used for the profile fit (clipped outside it)
RADIAL_BIN = 100
RADIAL_DEG = 3
INJ_TYPES = ['flare']
INJ_MIN_SNR = 3.0
INJ_MATCH_RADIUS = 1.5                 # centroid_match_radius used when gathering the injections
REAL_R_MAX = 2.0                       # px: sorted-flare offsets beyond this are dropped (likelihood truncated there)
SNR_REF = 20.0
NSIGMA = 3.0                           # CROSSMATCH_NSIGMA, for the report
N_BOOT = 50                            # bootstrap refits of the real-sky term (0 to skip)
OUT_DIR = f'{HERE}/localisation_calibration'
# ----------------

KEY = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
PNG = re.compile(r'^S(\d+)C(\d+)C(\d+)C(\d+)O(\d+)E(\d+)\.png$', re.IGNORECASE)


# ----------------------------- Geometry ----------------------------- #

def optical_axis():
    """{(camera, ccd): (x, y)}: the focal-plane origin in 0-based FFI pixels (column includes the 44 prescan)."""
    from tess_stars2px import Levine_FPG
    fpg = Levine_FPG()
    return {(c + 1, k + 1): tuple(float(v) for v in fpg.mm_to_pix_single_ccd(c, np.array([0., 0.]), k)[0] + [44, 0])
            for c in range(4) for k in range(4)}


def toward_axis(d, axis):
    """Distance to the optical axis and the unit vector toward it, per row of d (needs camera, ccd, xccd, yccd)."""
    ax = np.array([axis[(int(c), int(k))] for c, k in zip(d.camera, d.ccd)])
    vx, vy = ax[:, 0] - d.xccd.to_numpy(float), ax[:, 1] - d.yccd.to_numpy(float)
    r = np.hypot(vx, vy)
    return r, vx / r, vy / r


def radial_shift(coef, r):
    return np.polyval(coef, np.clip(r, *RADIAL_RANGE) / 1000)


# ----------------------------- Model ----------------------------- #

def log_rayleigh(r, s, rmax):
    return np.log(r) - 2 * np.log(s) - r ** 2 / (2 * s ** 2) - np.log1p(-np.exp(-rmax ** 2 / (2 * s ** 2)))


def eps_of(snr, e0, e1):
    return expit(e0 + e1 * np.log10(np.asarray(snr, float) / SNR_REF))


def mixture_nll(r, sig_core, sig_out, eps, rmax):
    if np.any(sig_out <= 1.5 * sig_core):          # keep the outlier component wider than the core
        return 1e12
    e = np.clip(eps, 1e-9, 1 - 1e-9)
    return -logsumexp([np.log1p(-e) + log_rayleigh(r, sig_core, rmax),
                       np.log(e) + log_rayleigh(r, sig_out, rmax)], axis=0).sum()


def _best(nll, starts):
    return min((minimize(nll, x0, method='Nelder-Mead',
                         options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-7, 'fatol': 1e-7}) for x0 in starts),
               key=lambda res: res.fun)


def fit_free(snr, r, rmax):
    """sigma = sqrt((a snr^-b)^2 + floor^2), all free, plus outliers."""
    def nll(th):
        a, b, fl, so = np.exp(th[:4])
        return mixture_nll(r, np.hypot(a * snr ** -b, fl), so, eps_of(snr, th[4], th[5]), rmax)
    starts = [list(np.log([a, 0.8, fl, so])) + [e0, 0.0] for a in (0.5, 1.5) for fl in (0.02, 0.05)
              for so in (0.3, 0.5) for e0 in (-3.5, -2.0)]
    res = _best(nll, starts)
    a, b, fl, so = np.exp(res.x[:4])
    return dict(a=a, b=b, floor=fl, sig_out=so, e0=res.x[4], e1=res.x[5]), res.fun


def fit_hybrid(snr, r, rmax, sig_inj, x0=None):
    """sigma = sqrt(sig_inj(snr)^2 + sys^2), only sys free (plus outliers)."""
    si = sig_inj(snr)

    def nll(th):
        sys_, so = np.exp(th[:2])
        return mixture_nll(r, np.hypot(si, sys_), so, eps_of(snr, th[2], th[3]), rmax)
    starts = [x0] if x0 is not None else [[np.log(s0), np.log(0.35), -2.5, -1.0] for s0 in (0.02, 0.04, 0.06)]
    res = _best(nll, starts)
    return dict(sys=np.exp(res.x[0]), sig_out=np.exp(res.x[1]), e0=res.x[2], e1=res.x[3]), res.fun, res.x


def sigma_ab(snr, a, b, floor):
    return np.hypot(a * np.asarray(snr, float) ** -b, floor)


# ----------------------------- Data ----------------------------- #

def sorted_labels(sort_dir):
    rows = []
    for group in os.listdir(sort_dir):
        gdir = os.path.join(sort_dir, group)
        if os.path.isdir(gdir):
            rows += [{**dict(zip(KEY, map(int, m.groups()))), 'label': group}
                     for m in map(PNG.match, os.listdir(gdir)) if m]
    lab = pd.DataFrame(rows).drop_duplicates()
    return lab[~lab.duplicated(KEY, keep=False)]      # events sorted into more than one group are dropped


def load_sorted():
    """Sorted flares with raw PSF-fit offsets from the pipeline's chosen star (event - star, px)."""
    ev = pd.concat([pd.read_csv(f, low_memory=False).assign(src=f) for f in EVENT_CSVS],
                   ignore_index=True).drop_duplicates(KEY)
    labels = (pd.read_csv(LABELS)[KEY + ['label']] if isinstance(LABELS, str)
              else pd.concat([sorted_labels(d) for d in LABELS], ignore_index=True))
    ev = ev.merge(labels, on=KEY, how='left')
    assumed = pd.Series(False, index=ev.index)
    for f, cams in ASSUME_FLARE.items():
        assumed |= (ev.src == f) & ev.camera.isin(cams) & ev.label.isna()
    ev.loc[assumed, 'label'] = LABEL
    unsorted = int((ev.label.isna() & ev.camera.isin(CAMERAS)).sum())
    s = ev[(ev.label == LABEL) & ev.camera.isin(CAMERAS) & (ev.psf_like > 0.5)].copy()
    num = lambda c: pd.to_numeric(s[c], errors='coerce')
    # nearest_gaia_dx/dy = star - event (event = xcentroid); raw PSF fit = xcentroid_psf (+ any y offset removed)
    s['dx'] = num('xcentroid_psf') - num('xcentroid') - num('nearest_gaia_dx')
    s['dy'] = num('ycentroid_psf') - num('ycentroid') - num('nearest_gaia_dy') + Y_OFFSET_IN_PSF
    s = s.dropna(subset=['dx', 'dy', 'snr_psf', 'xccd', 'yccd']).reset_index(drop=True)
    return s, int(assumed.sum()), unsorted


# ----------------------------- Main ----------------------------- #

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = []

    # 1. optical axis
    axis = optical_axis()
    xs = np.array([axis[(c, k)][0] for c in range(1, 5) for k in (1, 3)])
    xo = np.array([axis[(c, k)][0] for c in range(1, 5) for k in (2, 4)])
    ys = np.array([v[1] for v in axis.values()])
    rep += ['1. Optical axis (0-based FFI px): CCDs 1,3 x %.0f-%.0f; CCDs 2,4 x %.0f-%.0f; y %.0f-%.0f'
            % (xs.min(), xs.max(), xo.min(), xo.max(), ys.min(), ys.max())]

    # 2. radial shift
    cal = pd.read_csv(CALIB_CSV, low_memory=False)
    cal = cal[cal.variable == 0].dropna(subset=['dx', 'dy', 'xccd', 'yccd']).copy()
    cal['r'], ux, uy = toward_axis(cal, axis)
    cal['t'] = cal.dx * ux + cal.dy * uy
    cal['p'] = -cal.dx * uy + cal.dy * ux
    edges = np.arange(RADIAL_RANGE[0], RADIAL_RANGE[1] + 1, RADIAL_BIN)
    g = cal.groupby(pd.cut(cal.r, edges), observed=True).agg(r=('r', 'median'), t=('t', 'median'), n=('t', 'size'))
    coef = np.polyfit(g.r / 1000, g.t, RADIAL_DEG, w=np.sqrt(g.n))
    rr = np.array([300, 1000, 1500, 2000, 2500, 2900])
    rep += [f'\n2. Radial shift from {len(cal)} flare stars ({os.path.basename(CALIB_CSV)}):',
            '   toward the axis at r = ' + ', '.join(f'{x}: {v:.4f}' for x, v in zip(rr, radial_shift(coef, rr))) + ' px',
            f'   sideways part: median {cal.p.median():+.4f} px (should be ~0)']
    by_cam = cal.groupby(['camera', pd.cut(cal.r, [0, 800, 1400, 2000, 2600, 3200])], observed=True).t.median().unstack()
    rep += ['   by camera (median toward-axis shift, r bins in px):', by_cam.round(3).to_string()]

    # 3. injection core
    inj = pd.read_csv(INJ_CSV, low_memory=False)
    inj = inj[(inj.detected == 'y') & inj.event_type.isin(INJ_TYPES) & (inj.snr_psf >= INJ_MIN_SNR)].copy()
    rep += [f'\n3. Injections: {len(inj)} recovered ({", ".join(INJ_TYPES)}, snr_psf >= {INJ_MIN_SNR}); '
            f'median dx {inj.dx.median():+.4f}, dy {inj.dy.median():+.4f} px (removed)']
    inj['r'] = np.hypot(inj.dx - inj.dx.median(), inj.dy - inj.dy.median())
    inj = inj[inj.r <= INJ_MATCH_RADIUS]
    Pi, _ = fit_free(inj.snr_psf.to_numpy(float), inj.r.to_numpy(), INJ_MATCH_RADIUS)
    rep += ['   core: ' + '  '.join(f'{k} {v:.4f}' for k, v in Pi.items())]

    def sig_inj(x):
        return sigma_ab(x, Pi['a'], Pi['b'], Pi['floor'])

    # 4. real-sky term
    s, n_assumed, n_unsorted = load_sorted()
    s['r_ax'], ux, uy = toward_axis(s, axis)
    f = radial_shift(coef, s.r_ax.to_numpy())
    s['dxc'], s['dyc'] = s.dx - f * ux, s.dy - f * uy
    s['r'] = np.hypot(s.dxc, s.dyc)
    n_far = int((s.r > REAL_R_MAX).sum())
    s = s[s.r <= REAL_R_MAX].reset_index(drop=True)
    snr, r = s.snr_psf.to_numpy(float), s.r.to_numpy()
    Ph, nll_h, xh = fit_hybrid(snr, r, REAL_R_MAX, sig_inj)
    Pf, nll_f = fit_free(snr, r, REAL_R_MAX)
    floor = float(np.hypot(Pi['floor'], Ph['sys']))
    rep += [f'\n4. Sorted flares: {len(s)} (cameras {CAMERAS}; {n_assumed} unsorted assumed {LABEL}, {n_unsorted} '
            f'unsorted left out, {n_far} beyond {REAL_R_MAX} px dropped); snr_psf {snr.min():.1f}-{snr.max():.1f}',
            '   median residual after the radial shift, by CCD (dx / dy): ' + '  '.join(
                f'{c}: {s.dxc[s.ccd == c].median():+.3f}/{s.dyc[s.ccd == c].median():+.3f}' for c in sorted(s.ccd.unique())),
            f'   injections (+) sys:  -lnL {nll_h:9.2f} (4 par)   ' + '  '.join(f'{k} {v:.4f}' for k, v in Ph.items()),
            f'   all free (check):    -lnL {nll_f:9.2f} (6 par)   ' + '  '.join(f'{k} {v:.4f}' for k, v in Pf.items())]
    if N_BOOT:
        rng = np.random.default_rng(1)
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(s), len(s))
            p, _, _ = fit_hybrid(snr[i], r[i], REAL_R_MAX, sig_inj, x0=list(xh))
            boot.append(p['sys'])
        lo, hi = np.percentile(boot, [16, 84])
        rep += [f'   sys bootstrap 16-84%: {lo:.4f}-{hi:.4f}  ->  floor {np.hypot(Pi["floor"], lo):.4f}-'
                f'{np.hypot(Pi["floor"], hi):.4f}']

    def sig(x):
        return sigma_ab(x, Pi['a'], Pi['b'], floor)

    rows = [f'{x:6.1f}  {sig(x):.4f}  {NSIGMA * sig(x):.3f}  {sigma_ab(x, Pf["a"], Pf["b"], Pf["floor"]):.4f}  '
            f'{eps_of(x, Ph["e0"], Ph["e1"]):.3f}' for x in (2, 3, 4, 5, 6.4, 8, 10, 15, 20, 30, 50, 100, 200)]
    rep += ['\nFinal sigma(snr) = sqrt((A snr^-B)^2 + FLOOR^2) per axis:',
            '   snr   sigma   %.0f*sigma   free-fit sigma   outlier fraction' % NSIGMA] + ['  ' + x for x in rows]
    s['inside'] = s.r <= NSIGMA * sig(snr)
    rep += [f'\nSorted flares within {NSIGMA:.0f} sigma: {s.inside.mean():.3f}; by SNR bin:',
            s.groupby(pd.cut(s.snr_psf, [0, 10, 15, 25, 50, 1e9]), observed=True).inside.agg(['size', 'mean'])
            .round(3).to_string()]

    block = ['# ---- generated by development/localisation_calibration.py ----',
             f'LOCALISATION_A = {Pi["a"]:.4f}',
             f'LOCALISATION_B = {Pi["b"]:.4f}',
             f'LOCALISATION_FLOOR = {floor:.4f}      # injection floor {Pi["floor"]:.4f} (+) real-sky {Ph["sys"]:.4f}',
             'OPTICAL_AXIS_PX = {'] + [f'    ({c}, {k}): ({x:.1f}, {y:.1f}),' for (c, k), (x, y) in axis.items()] + [
             '}',
             'RADIAL_SHIFT_COEF = (' + ', '.join(f'{c:.6f}' for c in coef) + ')',
             f'RADIAL_SHIFT_RANGE = ({RADIAL_RANGE[0]}, {RADIAL_RANGE[1]})']
    text = '\n'.join(rep + ['', 'Constants for tessellate/localisation.py:'] + block)
    print(text)
    open(f'{OUT_DIR}/report.txt', 'w').write(text + '\n')

    # plots
    fig, ax = plt.subplots(ncols=3, figsize=(19, 5))
    a = ax[0]
    for cam, c in zip([1, 2, 3, 4], ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']):
        gc = cal[cal.camera == cam]
        m = gc.groupby(pd.cut(gc.r, np.arange(0, 3101, 200)), observed=True).agg(r=('r', 'median'), t=('t', 'median'))
        a.plot(m.r, m.t, 'o', ms=4, color=c, label=f'calibration flares, camera {cam}')
    grid = np.linspace(*RADIAL_RANGE, 200)
    a.plot(grid, radial_shift(coef, grid), 'k-', lw=2, label='fitted shift')
    a.axhline(0, color='0.5', lw=0.8)
    a.set_xlabel('distance from the optical axis (px)'); a.set_ylabel('offset toward the axis (px)')
    a.set_title('radial shift'); a.legend(fontsize=8)
    a = ax[1]
    grid = np.geomspace(2, 300, 200)
    bi = inj.groupby(pd.cut(inj.snr_psf, np.geomspace(3, 300, 16)), observed=True)
    a.plot(bi.snr_psf.median(), bi.r.median() / 1.1774, 'o', color='#1baf7a', ms=5,
           label='injections (binned, median r / 1.177)')
    bs = s.groupby(pd.cut(s.snr_psf, np.geomspace(6, 300, 11)), observed=True)
    a.plot(bs.snr_psf.median(), bs.r.median() / 1.1774, 's', color='#2a78d6', ms=6,
           label='sorted flares (binned, median r / 1.177)')
    a.plot(grid, sig_inj(grid), '-', color='#1baf7a', lw=2, label='injection core')
    a.plot(grid, sigma_ab(grid, Pf['a'], Pf['b'], Pf['floor']), '--', color='#2a78d6', lw=2, label='free fit, flares')
    a.plot(grid, sig(grid), '-', color='#eb6834', lw=2, label=f'final: injections (+) {Ph["sys"]:.3f} px')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('snr_psf'); a.set_ylabel('sigma per axis (px)')
    a.set_title('core width'); a.legend(fontsize=8)
    a = ax[2]
    a.scatter(s.snr_psf, s.r, s=3, alpha=0.2, color='0.4', rasterized=True, label='sorted flares')
    a.plot(grid, NSIGMA * sig(grid), '-', color='#eb6834', lw=2, label=f'{NSIGMA:.0f} sigma')
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.005, REAL_R_MAX)
    a.set_xlabel('snr_psf'); a.set_ylabel('radial offset (px)'); a.set_title('match radius'); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT_DIR}/calibration.png', dpi=110)


if __name__ == '__main__':
    main()
