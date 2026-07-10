"""
PSF flux calibration for tessellate reference images.

Fits PSF photometry on bright, unsaturated, isolated Gaia DR3 sources in the
reference image and derives a TESS-to-Gaia-Rp AB magnitude zeropoint.

Each star is fit simultaneously for flux, sub-pixel position (dx, dy), and a
flat background offset.  Stars are processed in parallel via joblib.

Usage
-----
from psf_flux_calibration import run_calibration
from astropy.io import fits
from astropy.wcs import WCS

with fits.open('Cam1/Ccd1/wcs/ref/corrected.fits') as f:
    image = f[1].data
    wcs = WCS(f[1].header)

cut_corners, _, _, _ = processor.find_cuts(cam=1, ccd=1, n=8, plot=False)

zp_ab, zp_err, df = run_calibration(
    image, wcs,
    sector=73, cam=1, ccd=1,
    cut_corner=cut_corners[cut - 1],
    plot=True, savepath='.'
)
"""

# Pin BLAS to a single thread per process BEFORE numpy imports.  Each joblib
# worker does its own linear algebra; without this, every worker spawns one BLAS
# thread per core, massively oversubscribing the node and starving CPU use.
import os
# Force single-threaded BLAS (overriding any SLURM-exported OMP_NUM_THREADS):
# parallelism here is across joblib processes, not BLAS threads.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings('ignore')

# Gaia DR3 Rp Vega -> AB offset, from synthetic photometry (pysynphot Vega
# spectrum through the actual Gaia3 Rp passband, via calibrimbore's
# get_pb_zpt): zp_AB - zp_Vega = 0.379. The previous literature value of
# 0.152 (Casagrande & VandenBerg 2018) was ~0.23 mag too small.
GAIA_RP_AB_OFFSET = 0.379


def _zp_to_scale(zp):
    """AB zeropoint (mag, a log quantity) -> linear flux-scale factor."""
    return 10 ** (0.4 * np.asarray(zp, dtype=float))


def _scale_to_zp(f):
    """Linear flux-scale factor -> AB zeropoint (mag)."""
    return 2.5 * np.log10(f)

TESS_PIX_SCALE = 21.0  # arcsec/pixel
GAIA_PATH_DEFAULT = '/fred/oz335/GAIAdata/full_gaia_cat.csv'
PRF_PATH_DEFAULT = '/fred/oz335/_local_TESS_PRFs'


# ---------------------------------------------------------------------------
# Gaia query
# ---------------------------------------------------------------------------

def _query_gaia(ra_centre, dec_centre, radius_arcsec, gaia_path, gmag_limit):
    """
    Cone-search the local Gaia catalogue.

    An RA/Dec bounding box and the Gmag cut are applied inside the scan so the
    haversine is only evaluated on the rows near the field.
    """
    import duckdb

    radius_deg = radius_arcsec / 3600.0
    dec_lo = dec_centre - radius_deg
    dec_hi = dec_centre + radius_deg
    cosd = max(float(np.cos(np.radians(dec_centre))), 1e-3)
    ra_pad = radius_deg / cosd

    # RA bounding box, handling wraparound at 0/360 deg
    ra_lo = ra_centre - ra_pad
    ra_hi = ra_centre + ra_pad
    if ra_lo < 0 or ra_hi > 360:
        ra_cond = f"(ra >= {ra_lo % 360} OR ra <= {ra_hi % 360})"
    else:
        ra_cond = f"(ra BETWEEN {ra_lo} AND {ra_hi})"

    return duckdb.sql(f"""
        SELECT * FROM (
            SELECT *,
                2 * degrees(asin(sqrt(
                    pow(sin(radians(("dec" - {dec_centre}) / 2)), 2) +
                    cos(radians("dec")) * cos(radians({dec_centre})) *
                    pow(sin(radians((ra - {ra_centre}) / 2)), 2)
                ))) * 3600 AS dist_arcsec
            FROM read_csv('{gaia_path}', ignore_errors=true)
            WHERE "dec" BETWEEN {dec_lo} AND {dec_hi}
              AND {ra_cond}
              AND Gmag < {gmag_limit}
        )
        WHERE dist_arcsec < {radius_arcsec}
    """).df()


def _parallel_map(worker, tasks, n_jobs, label, batch=64):
    """
    Run `worker(*task)` over `tasks` in parallel, in batches, printing progress
    to stdout (flushed) after each batch so a long run shows real progress in
    the job output log -- and pinpoints where it stalls if it does.
    """
    from joblib import Parallel, delayed
    n = len(tasks)
    out = []
    _t0 = time.time()
    for i in range(0, n, batch):
        chunk = tasks[i:i + batch]
        out.extend(Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(worker)(*a) for a in chunk))
        done = min(i + batch, n)
        n_ok = sum(r is not None for r in out)
        print(f'    {label}: {done}/{n} processed '
              f'({n_ok} ok, {time.time() - _t0:.1f}s)', flush=True)
    return out


# ---------------------------------------------------------------------------
# Star selection
# ---------------------------------------------------------------------------

def _select_isolated(gaia_all, gaia_cal, wcs, cut_corner, image_shape,
                     iso_radius_pix, edge_margin, delta_mag=2.0):
    """
    Return calibration stars that:
      - fall within the cut image (with edge_margin buffer)
      - have no Gaia neighbour within iso_radius_pix TESS pixels that is
        brighter than star_rp + delta_mag

    The WCS is the full CCD WCS; cut_corner=(x0,y0) gives the CCD pixel
    coordinate of the bottom-left corner of the cut image.
    """
    ny, nx = image_shape
    x0, y0 = cut_corner
    iso_arcsec = iso_radius_pix * TESS_PIX_SCALE

    ccd_x, ccd_y = wcs.all_world2pix(gaia_cal.ra.values, gaia_cal.dec.values, 0)
    loc_x = ccd_x - x0
    loc_y = ccd_y - y0

    ra_all = gaia_all.ra.values
    dec_all = gaia_all.dec.values
    rp_all = gaia_all.RPmag.values

    keep = []
    for i in range(len(gaia_cal)):
        lx, ly = loc_x[i], loc_y[i]
        if not (edge_margin < lx < nx - edge_margin and
                edge_margin < ly < ny - edge_margin):
            continue
        cos_dec = np.cos(np.radians(gaia_cal.dec.values[i]))
        dra = (ra_all - gaia_cal.ra.values[i]) * cos_dec
        ddec = dec_all - gaia_cal.dec.values[i]
        sep = np.sqrt(dra**2 + ddec**2) * 3600.0
        mag_limit = gaia_cal.RPmag.values[i] + delta_mag
        close = (sep > 0.5) & (sep < iso_arcsec) & (rp_all < mag_limit)
        if np.sum(close) == 0:
            keep.append(i)

    keep = np.array(keep, dtype=int)
    if len(keep) == 0:
        return gaia_cal.iloc[:0].reset_index(drop=True), np.array([]), np.array([]), np.array([]), np.array([])

    return (gaia_cal.iloc[keep].reset_index(drop=True),
            ccd_x[keep], ccd_y[keep],
            loc_x[keep], loc_y[keep])


# ---------------------------------------------------------------------------
# Cached PRF construction
# ---------------------------------------------------------------------------

# Module-level cache, persistent within each (loky) worker process.  The TESS
# PRF varies slowly across the CCD, so positions are bucketed to a coarse grid
# and one PRF is built per bucket instead of one per star.
_PRF_CACHE = {}


def _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir, bucket=100):
    from PRF import TESS_PRF
    col = int(np.clip(ccd_x, 44, 2090))
    row = int(np.clip(ccd_y, 1, 2040))
    cb = int(np.clip((col // bucket) * bucket + bucket // 2, 44, 2090))
    rb = int(np.clip((row // bucket) * bucket + bucket // 2, 1, 2040))
    key = (cam, ccd, sector, cb, rb, prf_dir)
    prf = _PRF_CACHE.get(key)
    if prf is None:
        prf = TESS_PRF(cam=cam, ccd=ccd, sector=sector,
                       colnum=cb, rownum=rb, localdatadir=prf_dir)
        _PRF_CACHE[key] = prf
    return prf


# ---------------------------------------------------------------------------
# Per-star PSF fit worker  (top-level so joblib can pickle it)
# ---------------------------------------------------------------------------

def _fit_star_worker(stamp, ccd_x, ccd_y, cam, ccd, sector, prf_dir,
                     stamp_size, ra, dec, loc_x, loc_y, rp_vega, bp_rp=np.nan):
    """
    Fit one calibration star.  Constructs TESS_PRF internally so the object
    does not need to cross process boundaries.

    Returns a dict of results, or None on failure.
    """
    cent = stamp_size // 2

    # Per-pixel noise from stamp corners
    c = 2
    corners = np.concatenate([stamp[:c, :c].ravel(), stamp[:c, -c:].ravel(),
                               stamp[-c:, :c].ravel(), stamp[-c:, -c:].ravel()])
    bg0 = np.nanmedian(corners)
    sigma_bg = np.nanstd(corners)
    if not np.isfinite(sigma_bg) or sigma_bg <= 0:
        sigma_bg = max(np.sqrt(abs(bg0)), 1.0)
    var_pix = sigma_bg**2

    flux0 = max(float(np.nansum(stamp - bg0)), 1.0)

    try:
        prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)
    except Exception:
        return None

    def _model(params):
        flux, dx, dy, bg = params
        p = prf.locate(cent + dx, cent + dy, (stamp_size, stamp_size))
        p = p / np.nansum(p)
        return flux * p + bg

    def _chi2(params):
        return float(np.nansum((stamp - _model(params))**2) / var_pix)

    try:
        res = minimize(_chi2, [flux0, 0.0, 0.0, bg0], method='BFGS',
                       options={'maxiter': 10000, 'gtol': 1e-6})
        flux, dx, dy, bg = res.x
        if not (flux > 0 and np.isfinite(flux)):
            return None
    except Exception:
        return None

    # Model and residual at the best fit
    p_fit = prf.locate(cent + float(dx), cent + float(dy), (stamp_size, stamp_size))
    p_fit = p_fit / np.nansum(p_fit)
    model_stamp = float(flux) * p_fit + float(bg)
    residual_stamp = stamp - model_stamp

    # Flux uncertainty from the analytic linear least-squares solution.
    # The model is linear in (flux, bg): data = flux * P + bg, with P the
    # normalised PSF.  The parameter covariance is s2 * inv(A^T A) for design
    # columns A = [P, 1], giving flux variance s2 * S11 / (Spp*S11 - Sp1^2).
    #
    # The per-pixel variance s2 is estimated from the fit residuals (not the
    # corners), so that an imperfect PSF model -- which biases the flux --
    # inflates the reported error.  This is the standard residual-variance
    # estimator (equivalent to scaling the covariance by reduced chi^2).
    #
    # Restrict to the inner 5x5 region around the centre, where essentially all
    # of the source flux and PSF-model power lies.
    inner = 2  # half-width of the 5x5 region
    inner_mask = np.zeros_like(stamp, dtype=bool)
    inner_mask[cent - inner:cent + inner + 1, cent - inner:cent + inner + 1] = True
    finite_pix = (inner_mask & np.isfinite(stamp) &
                  np.isfinite(p_fit) & np.isfinite(residual_stamp))
    P = p_fit[finite_pix]
    Spp = float(np.sum(P * P))
    Sp1 = float(np.sum(P))
    S11 = float(P.size)
    det = Spp * S11 - Sp1**2

    n_params = 4  # flux, dx, dy, bg
    dof = P.size - n_params
    if dof > 0:
        s2 = float(np.sum(residual_stamp[finite_pix]**2) / dof)
    else:
        s2 = var_pix
    if det > 0 and s2 > 0:
        e_flux = float(np.sqrt(s2 * S11 / det))
    else:
        e_flux = np.nan

    rp_ab = rp_vega + GAIA_RP_AB_OFFSET
    zp = rp_ab + 2.5 * np.log10(flux)
    e_zp = (2.5 / np.log(10)) * (e_flux / flux) if (np.isfinite(e_flux) and flux > 0) else np.nan

    return {
        'ra': ra, 'dec': dec,
        'x_pix': loc_x, 'y_pix': loc_y,
        'dx_fit': float(dx), 'dy_fit': float(dy),
        'flux_psf': float(flux), 'e_flux_psf': e_flux,
        'background': float(bg),
        'rp_vega': rp_vega, 'rp_ab': rp_ab,
        'bp_rp': bp_rp,
        'zp_ab': zp, 'e_zp_ab': e_zp,
        'stamp_data': stamp,
        'model_data': model_stamp,
        'residual_data': residual_stamp,
    }


# ---------------------------------------------------------------------------
# Stage-2 scene fit worker  (shared ZP centre, per-source variability freedom)
# ---------------------------------------------------------------------------

def _scene_fit_worker(stamp, local_x, local_y, flux_lo, flux_hi, flux0,
                      pos_tol_x, pos_tol_y, target_k,
                      ccd_x, ccd_y, cam, ccd, sector, prf_dir, stamp_size,
                      ra, dec, rp_ab, bp_rp, loc_x, loc_y):
    """
    Fit one calibrator together with every catalogued neighbour overlapping its
    stamp.  The single cut-wide zeropoint sets the CENTRE of each source's flux
    bound (flux_lo/flux_hi = ZP-predicted flux over the allowed magnitude
    tolerance), but each flux is free within that window so stellar variability
    (and catalogue/colour error) is absorbed rather than forced onto the residual.

    Positions: only the TARGET may shift, by at most +/- (pos_tol_x, pos_tol_y)
    pixels from its catalogue position (the tolerance is the stage-1 position
    scatter); all neighbours are held fixed.  Fitting just the target keeps the
    problem a fast 2-D search with a linear inner flux solve, and -- because
    neighbours cannot move -- removes any chance of a faint source being pulled
    toward a bright neighbour.  If the tolerance is zero, positions are fixed and
    the scene is a single bounded linear least-squares solve.

    Error region: the inner 5x5 pixels around the target.
    Returns a dict for the target source, or None.
    """
    from scipy.optimize import lsq_linear

    npix = stamp_size * stamp_size
    cent = stamp_size // 2
    inner = 2  # half-width of the inner 5x5 error region

    try:
        prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)
    except Exception:
        return None

    K = len(local_x)
    data = stamp.ravel()
    finite = np.isfinite(data)
    if finite.sum() < K + 1:
        return None

    def _psf_col(lx, ly):
        p = prf.locate(lx, ly, (stamp_size, stamp_size))
        s = np.nansum(p)
        if not np.isfinite(s) or s <= 0:
            return np.zeros(npix)
        return (p / s).ravel()

    pos_tol_x = np.asarray(pos_tol_x, dtype=float)
    pos_tol_y = np.asarray(pos_tol_y, dtype=float)
    fit_positions = bool(np.any(pos_tol_x > 0) or np.any(pos_tol_y > 0))

    ones = np.ones(npix)
    flux_bounds = (np.concatenate([flux_lo, [-np.inf]]),
                   np.concatenate([flux_hi, [np.inf]]))

    # Neighbour + background columns are fixed; only the target may move
    fixed_cols = [None] * K
    for k in range(K):
        if k != target_k:
            fixed_cols[k] = _psf_col(local_x[k], local_y[k])

    def _build_A(dx, dy):
        cols = [fixed_cols[k] if k != target_k
                else _psf_col(local_x[k] + dx, local_y[k] + dy)
                for k in range(K)]
        return np.column_stack(cols + [ones])

    tol_x = float(np.max(pos_tol_x)) if pos_tol_x.size else 0.0
    tol_y = float(np.max(pos_tol_y)) if pos_tol_y.size else 0.0

    if fit_positions and (tol_x > 0 or tol_y > 0):
        # ---- Bounded TARGET position only: 2-D search, linear inner solve ----
        from scipy.optimize import minimize
        df_idx = finite

        def _chi2(dxy):
            A = _build_A(dxy[0], dxy[1])
            Ag = A[df_idx]
            if not np.all(np.isfinite(Ag)):
                return 1e30
            sol, *_ = np.linalg.lstsq(Ag, data[df_idx], rcond=None)
            r = Ag @ sol - data[df_idx]
            return float(r @ r)

        try:
            opt = minimize(_chi2, [0.0, 0.0], method='L-BFGS-B',
                           bounds=[(-tol_x, tol_x), (-tol_y, tol_y)])
            dx_t, dy_t = float(opt.x[0]), float(opt.x[1])
        except Exception:
            dx_t = dy_t = 0.0
    else:
        dx_t = dy_t = 0.0

    # Final bounded flux solve at the chosen position
    A = _build_A(dx_t, dy_t)
    good = finite & np.all(np.isfinite(A), axis=1)
    if good.sum() < K + 1:
        return None
    try:
        res = lsq_linear(A[good], data[good], bounds=flux_bounds,
                         method='trf', max_iter=200)
    except Exception:
        return None
    fluxes = res.x[:K]
    bg = float(res.x[K])
    Acov = A[good]

    flux = float(fluxes[target_k])
    if not (flux > 0 and np.isfinite(flux)):
        return None

    model_stamp = (A @ np.concatenate([fluxes, [bg]])).reshape(stamp_size, stamp_size)
    residual_stamp = stamp - model_stamp

    # Per-pixel variance from the inner-region residuals (PSF mismatch inflates it)
    inner_mask = np.zeros((stamp_size, stamp_size), dtype=bool)
    inner_mask[cent - inner:cent + inner + 1, cent - inner:cent + inner + 1] = True
    inner_good = inner_mask & np.isfinite(residual_stamp)
    dof_inner = int(inner_good.sum()) - (K + 1)
    if dof_inner > 0:
        s2 = float(np.sum(residual_stamp[inner_good]**2) / dof_inner)
    else:
        s2 = float(np.sum(residual_stamp[np.isfinite(residual_stamp)]**2) /
                   max(int(finite.sum()) - (K + 1), 1))

    # Flux covariance (positions held at solution), scaled by the local variance
    try:
        cov = s2 * np.linalg.inv(Acov.T @ Acov)
        e_flux = float(np.sqrt(cov[target_k, target_k]))
    except Exception:
        e_flux = np.nan

    zp = rp_ab + 2.5 * np.log10(flux)
    e_zp = (2.5 / np.log(10)) * (e_flux / flux) if (np.isfinite(e_flux) and flux > 0) else np.nan

    # Flag if the target flux railed against (or sits within its error of) the
    # flux bound -- such a point is set by the prior, not the data.
    lo_t = float(flux_lo[target_k])
    hi_t = float(flux_hi[target_k])
    ef = e_flux if np.isfinite(e_flux) else 0.0
    at_bound = bool(flux - ef <= lo_t or flux + ef >= hi_t)

    return {
        'ra': ra, 'dec': dec,
        'x_pix': loc_x, 'y_pix': loc_y,
        'dx_fit': dx_t, 'dy_fit': dy_t,
        'flux_psf': flux, 'e_flux_psf': e_flux,
        'background': bg,
        'rp_vega': rp_ab - GAIA_RP_AB_OFFSET, 'rp_ab': rp_ab,
        'bp_rp': bp_rp,
        'zp_ab': zp, 'e_zp_ab': e_zp,
        'n_scene': int(K),
        'at_bound': at_bound,
        'stamp_data': stamp,
        'model_data': model_stamp,
        'residual_data': residual_stamp,
    }


# ---------------------------------------------------------------------------
# Stage-2 ZP refinement via scene modelling (single cut-wide zeropoint)
# ---------------------------------------------------------------------------

def _refine_zeropoint_scene(image, wcs, gaia_deep, cut_corner, cam, ccd, sector,
                            prf_dir, stamp_size, mag_lo, mag_hi, edge_margin,
                            n_jobs, zp_ab0, zp_err0,
                            pos_tol_x=0.0, pos_tol_y=0.0,
                            var_tol_mag=0.5, refine_iter=3, refine_tol=1e-3,
                            max_err_factor=3.0, max_zp_err=0.1,
                            zp_bin_width=0.5, zp_bin_min_n=5):
    """
    Stage 2: drop the isolation cut and refine a single zeropoint that is
    constant across the cut and common to every scene.

    The zeropoint sets the centre of each source's flux bound, but every flux is
    free within +/- a magnitude tolerance (max of 3*sigma_scatter and
    var_tol_mag) so stellar variability is absorbed.  Each iteration fits all
    candidate stamps in parallel, then recomputes the 3-sigma-clipped global ZP;
    this block-coordinate scheme (parallel per-stamp fit <-> global ZP update)
    converges to a joint fit with one shared ZP.

    Returns (zp_ab, zp_err, df) or (None, None, None) if it cannot proceed.
    """
    from joblib import Parallel, delayed

    ny, nx = image.shape
    x0, y0 = cut_corner
    half = stamp_size // 2

    ccd_x_all, ccd_y_all = wcs.all_world2pix(gaia_deep.ra.values,
                                             gaia_deep.dec.values, 0)
    loc_x_all = ccd_x_all - x0
    loc_y_all = ccd_y_all - y0
    rp_ab_all = gaia_deep.RPmag.values + GAIA_RP_AB_OFFSET
    has_bp = 'BPmag' in gaia_deep.columns
    bp_all = ((gaia_deep.BPmag.values - gaia_deep.RPmag.values)
              if has_bp else np.full(len(gaia_deep), np.nan))

    # Candidate calibrators: in-band and on-image, but NO isolation requirement
    cand = np.where((rp_ab_all >= mag_lo) & (rp_ab_all <= mag_hi) &
                    (loc_x_all > edge_margin) & (loc_x_all < nx - edge_margin) &
                    (loc_y_all > edge_margin) & (loc_y_all < ny - edge_margin))[0]

    # Static per-candidate setup (stamp + neighbour list); reused every iteration
    base = []
    for ci in cand:
        lx_int = int(np.round(loc_x_all[ci]))
        ly_int = int(np.round(loc_y_all[ci]))
        sy0, sy1 = ly_int - half, ly_int + half + 1
        sx0, sx1 = lx_int - half, lx_int + half + 1
        if sy0 < 0 or sx0 < 0 or sy1 > ny or sx1 > nx:
            continue
        stamp = image[sy0:sy1, sx0:sx1].astype(float).copy()
        if not np.isfinite(stamp).all():
            continue
        in_stamp = np.where((loc_x_all >= sx0 - 0.5) & (loc_x_all < sx1 + 0.5) &
                            (loc_y_all >= sy0 - 0.5) & (loc_y_all < sy1 + 0.5))[0]
        tk = int(np.where(in_stamp == ci)[0][0])
        base.append({
            'stamp': stamp, 'in_stamp': in_stamp,
            'lx_local': loc_x_all[in_stamp] - sx0,
            'ly_local': loc_y_all[in_stamp] - sy0,
            'rp_src': rp_ab_all[in_stamp],
            'tk': tk,
            'ccd_x': ccd_x_all[ci], 'ccd_y': ccd_y_all[ci],
            'ra': float(gaia_deep.ra.values[ci]),
            'dec': float(gaia_deep.dec.values[ci]),
            'rp_ab': float(rp_ab_all[ci]), 'bp_rp': float(bp_all[ci]),
            'loc_x': float(loc_x_all[ci]), 'loc_y': float(loc_y_all[ci]),
        })

    if len(base) < 2:
        print('  Scene refinement skipped: too few on-image candidates.')
        return None, None, None, None, None

    pos_msg = (f'positions free within +/-({pos_tol_x:.3f},{pos_tol_y:.3f}) px'
               if (pos_tol_x > 0 or pos_tol_y > 0) else 'positions fixed')
    print(f'  Scene refinement: {len(base)} candidates '
          f'(isolation cut dropped, {pos_msg})')

    zp = float(zp_ab0)
    ze = float(zp_err0) if np.isfinite(zp_err0) else var_tol_mag
    df2 = None
    for it in range(refine_iter):
        # Per-source flux window: ZP-predicted flux over the variability tolerance
        tol = max(3.0 * ze, var_tol_mag)
        args_list = []
        for t in base:
            f_lo = 10 ** ((zp - tol - t['rp_src']) / 2.5)
            f_hi = 10 ** ((zp + tol - t['rp_src']) / 2.5)
            f0 = 10 ** ((zp - t['rp_src']) / 2.5)
            ns = len(t['rp_src'])
            ptx = np.full(ns, pos_tol_x)
            pty = np.full(ns, pos_tol_y)
            args_list.append((
                t['stamp'], t['lx_local'], t['ly_local'], f_lo, f_hi, f0,
                ptx, pty, t['tk'],
                t['ccd_x'], t['ccd_y'], cam, ccd, sector, prf_dir, stamp_size,
                t['ra'], t['dec'], t['rp_ab'], t['bp_rp'], t['loc_x'], t['loc_y'],
            ))

        print(f'  [refine {it+1}/{refine_iter}] fitting {len(args_list)} scenes '
              f'(n_jobs={n_jobs}) ...', flush=True)
        _t = time.time()
        results = _parallel_map(_scene_fit_worker, args_list, n_jobs,
                                f'refine {it+1}')
        rows = [r for r in results if r is not None]
        if len(rows) < 2:
            print(f'  [refine {it+1}] too few successful scene fits; stopping.')
            break

        df2 = pd.DataFrame(rows)
        # Drop scenes pinned to the flux bound (set by the prior, not the data)
        # and poorly-constrained scenes (large error).
        n_raw = len(df2)
        n_bound = int(np.sum(df2.at_bound.values))
        keep = (_err_keep_mask(df2.e_zp_ab.values, max_err_factor, max_zp_err)
                & ~df2.at_bound.values)
        df2 = df2[keep].reset_index(drop=True)
        if len(df2) < 2:
            print(f'  [refine {it+1}] too few well-constrained scenes; stopping.')
            break
        # Empirical magnitude-binned robust combine (handles the faint wing)
        zp_new, ze_new, scat_new, bins = _binned_zeropoint(
            df2.zp_ab.values, df2.rp_ab.values, zp_bin_width, zp_bin_min_n)
        dzp = abs(zp_new - zp)
        print(f'  [refine {it+1}/{refine_iter}] zp={zp_new:.4f} +/- {ze_new:.4f} '
              f'(scatter={scat_new:.3f})  (N={len(df2)}/{n_raw}, {n_bound} at bound, '
              f'{len(bins)} bins, tol={tol:.3f} mag, dZP={dzp:.5f}, '
              f'{time.time() - _t:.1f}s)')

        zp = zp_new
        ze = ze_new
        if dzp < refine_tol:
            print(f'  Scene refinement converged after {it+1} iteration(s).')
            break

    if df2 is None:
        return None, None, None, None, None

    zp_ab, zp_err, zp_scatter, bins = _binned_zeropoint(
        df2.zp_ab.values, df2.rp_ab.values, zp_bin_width, zp_bin_min_n)
    print(f'  Global ZP (scene): {zp_ab:.4f} +/- {zp_err:.4f} mag '
          f'(core scatter {zp_scatter:.3f}, N={len(df2)} stamps, {len(bins)} mag bins)')
    return zp_ab, zp_err, zp_scatter, df2, bins


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(image, wcs, sector, cam, ccd,
                    cut_corner=(0, 0),
                    gaia_path=GAIA_PATH_DEFAULT,
                    prf_path=PRF_PATH_DEFAULT,
                    mag_lo=11.0, mag_hi=15.5,
                    iso_radius_pix=4.0, stamp_size=9,
                    delta_mag=2.0, edge_margin=5,
                    scene_refine=True, scene_maglim=16.5,
                    var_tol_mag=0.5, pos_flex=True, pos_nsig=3.0,
                    refine_iter=3, refine_tol=1e-3,
                    max_err_factor=3.0, max_zp_err=0.1,
                    zp_bin_width=0.5, zp_bin_min_n=5,
                    n_jobs=-1,
                    plot=True, savepath=None):
    """
    Derive a TESS-to-Gaia-Rp AB zeropoint from PSF photometry on a reference image.

    Parameters
    ----------
    image : 2D ndarray
        Reference image for the cut (background fitted per-star).
    wcs : astropy.wcs.WCS
        Full CCD WCS from wcs/ref/corrected.fits.  Calling
        wcs.all_world2pix(ra, dec, 0) returns CCD-level pixel coordinates.
    sector, cam, ccd : int
        TESS identifiers.
    cut_corner : (x0, y0)
        CCD pixel coordinate of the bottom-left corner of this cut, taken
        directly from cut_corners[cut-1] returned by DataProcessor.find_cuts.
    gaia_path : str
        Path to the local Gaia DR3 CSV file.
    prf_path : str
        Root directory of local TESS PRF files.
    mag_lo, mag_hi : float
        Gaia Rp (Vega) magnitude range for calibration stars.
    iso_radius_pix : float
        Isolation radius in TESS pixels.
    stamp_size : int (odd)
        Pixel width of the stamp cutout for each PSF fit.
    delta_mag : float
        A calibration star is rejected if any Gaia neighbour within
        iso_radius_pix pixels is brighter than star_rp + delta_mag.
    edge_margin : int
        Minimum pixel distance from cut-image edge for a usable star.
    n_jobs : int
        Number of parallel workers passed to joblib.Parallel.
        -1 uses all available cores.
    plot : bool
        If True, write diagnostic figures to savepath.
    savepath : str or None
        Directory for output files.

    Returns
    -------
    zp_ab : float
        AB zeropoint: Rp_AB = -2.5*log10(flux) + zp_ab.  Stage 1 uses a
        3-sigma-clipped mean; stage 2 uses the empirical magnitude-binned
        robust combine (median per Rp_AB bin, inverse-variance weighted by the
        per-bin scatter).
    zp_err : float
        Uncertainty on the combined zeropoint (stage 1: clipped std; stage 2:
        standard error of the magnitude-binned combine).
    results : pd.DataFrame
        Full per-star table.
    Writes (if savepath is set):
        psf_calibration_zp.csv    — zp_ab, e_zp_ab, n_stars
        psf_calibration_stars.csv — x, y, ra, dec, flux, e_flux, background, Rp_ab
    """
    from joblib import Parallel, delayed

    _t_start = time.time()
    print(f'=== PSF flux calibration: Sector{sector} Cam{cam} Ccd{ccd} '
          f'cut_corner={cut_corner} ===')

    ny, nx = image.shape
    x0, y0 = cut_corner

    # Field centre from the cut image centre in CCD coordinates
    ra_cen, dec_cen = wcs.all_pix2world(x0 + nx / 2.0, y0 + ny / 2.0, 0)
    radius_arcsec = np.hypot(nx / 2.0, ny / 2.0) * TESS_PIX_SCALE * 1.1

    print(f'Field centre: RA={ra_cen:.4f} Dec={dec_cen:.4f}')
    print(f'Querying Gaia DR3 within {radius_arcsec/60:.1f} arcmin ...')

    # Query deep enough to catch all relevant neighbours.  Stage 1 needs the
    # faintest cal star + delta_mag; stage 2 scene fitting needs the full source
    # list down to scene_maglim so neighbour flux is not dumped into targets.
    query_maglim = mag_hi + delta_mag
    if scene_refine:
        query_maglim = max(query_maglim, scene_maglim)
    _t = time.time()
    gaia_all = _query_gaia(ra_cen, dec_cen, radius_arcsec, gaia_path, query_maglim)
    print(f'  {len(gaia_all)} sources (Gmag < {query_maglim})  '
          f'[Gaia query {time.time() - _t:.1f}s]')

    if 'RPmag' not in gaia_all.columns:
        raise RuntimeError("'RPmag' column not found in Gaia catalog.")

    # Selection limits apply to the AB-converted Gaia Rp magnitude
    gaia_cal = gaia_all.dropna(subset=['RPmag']).copy()
    rp_ab_all = gaia_cal.RPmag.values + GAIA_RP_AB_OFFSET
    gaia_cal = gaia_cal[(rp_ab_all >= mag_lo) & (rp_ab_all <= mag_hi)].copy()
    print(f'  {len(gaia_cal)} candidates with {mag_lo} <= Rp_AB <= {mag_hi}')

    if len(gaia_cal) == 0:
        raise RuntimeError(f'No Gaia stars with {mag_lo} <= Rp_AB <= {mag_hi} in field.')

    gaia_iso, ccd_xs, ccd_ys, loc_xs, loc_ys = _select_isolated(
        gaia_all, gaia_cal, wcs, cut_corner, image.shape, iso_radius_pix, edge_margin, delta_mag)
    print(f'  {len(gaia_iso)} isolated, on-image calibration stars')

    if len(gaia_iso) == 0:
        raise RuntimeError('No isolated calibration stars remain after cuts.')

    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    half = stamp_size // 2

    has_bp = 'BPmag' in gaia_iso.columns

    # Build per-star argument list; extract stamps here (serial, cheap)
    tasks = []
    for i in range(len(gaia_iso)):
        lx_int = int(np.round(loc_xs[i]))
        ly_int = int(np.round(loc_ys[i]))
        sy0, sy1 = ly_int - half, ly_int + half + 1
        sx0, sx1 = lx_int - half, lx_int + half + 1
        if sy0 < 0 or sx0 < 0 or sy1 > ny or sx1 > nx:
            continue
        stamp = image[sy0:sy1, sx0:sx1].copy().astype(float)
        if not np.isfinite(stamp).all():
            continue
        bp_rp = (float(gaia_iso.BPmag.values[i]) - float(gaia_iso.RPmag.values[i])
                 if has_bp else np.nan)
        tasks.append((
            stamp,
            ccd_xs[i], ccd_ys[i],
            cam, ccd, sector, prf_dir,
            stamp_size,
            float(gaia_iso.ra.values[i]),
            float(gaia_iso.dec.values[i]),
            float(loc_xs[i]), float(loc_ys[i]),
            float(gaia_iso.RPmag.values[i]),
            bp_rp,
        ))

    print(f'  Stage 1: fitting {len(tasks)} isolated stars '
          f'(n_jobs={n_jobs}) ...', flush=True)

    _t = time.time()
    results_raw = _parallel_map(_fit_star_worker, tasks, n_jobs, 'stage 1')

    rows = [r for r in results_raw if r is not None]

    if len(rows) == 0:
        raise RuntimeError('PSF fitting failed for all calibration stars.')

    df = pd.DataFrame(rows)
    print(f'  {len(df)} successful PSF fits  [stage 1 fit {time.time() - _t:.1f}s]')

    # Drop poorly-constrained fits (large error) before anything downstream
    n_before = len(df)
    df = df[_err_keep_mask(df.e_zp_ab.values, max_err_factor,
                           max_zp_err)].reset_index(drop=True)
    if len(df) < n_before:
        print(f'  Dropped {n_before - len(df)} stars with large errors '
              f'(> {max_err_factor:g}x median or > {max_zp_err:g} mag)')

    # Sigma-clipped zeropoint (sigma=3)
    finite = np.isfinite(df.zp_ab.values)
    if finite.sum() < 2:
        raise RuntimeError('Fewer than 2 stars with finite zeropoints.')

    clip_mask = _sigma_clip_mask(df.zp_ab.values[finite], nsigma=3)
    zp_vals_clipped = df.zp_ab.values[finite][clip_mask]
    # zp_ab is a log (magnitude) quantity -- a flux ratio in disguise -- so the
    # combine must happen in linear flux space, not by averaging magnitudes
    # directly (that gives the geometric, not arithmetic, mean of the flux
    # scale and is biased relative to it).
    zp_ab = float(_scale_to_zp(np.mean(_zp_to_scale(zp_vals_clipped))))
    zp_err = float(np.std(zp_vals_clipped))

    print(f'\nStage 1 Zeropoint (AB): {zp_ab:.4f} +/- {zp_err:.4f} mag  '
          f'(N={len(zp_vals_clipped)} isolated stars, 3-sigma clipped mean)')
    print(f'  formula: Rp_AB = -2.5 * log10(flux) + {zp_ab:.4f}')
    print(f'  (Gaia Rp Vega-to-AB offset: +{GAIA_RP_AB_OFFSET:.3f} mag, '
          'Casagrande & VandenBerg 2018)')

    # Keep the stage-1 (isolated-star) solution for the comparison figure
    df_stage1 = df
    zp_stage1, ze_stage1 = zp_ab, zp_err
    zp_scatter = zp_err          # stage-1 std is its scatter; stage 2 overrides
    scene_bins = None
    stage2_done = False

    # Stage 2: scene-model refinement using the full (un-isolated) source list.
    # Positions are held fixed (no per-source freedom); fluxes float within a
    # magnitude tolerance about the single cut-wide ZP to absorb variability.
    if scene_refine:
        print('\nStage 2: scene-model ZP refinement ...')
        # Position freedom for stage 2 = stage-1 sub-pixel offset scatter.
        # This bounds how far a faint source can be pulled toward a bright one.
        pos_tol_x = pos_tol_y = 0.0
        if pos_flex:
            dxv = df_stage1.dx_fit.values
            dyv = df_stage1.dy_fit.values
            mdx = _sigma_clip_mask(dxv, nsigma=3) if np.isfinite(dxv).sum() >= 2 else np.ones(len(dxv), bool)
            mdy = _sigma_clip_mask(dyv, nsigma=3) if np.isfinite(dyv).sum() >= 2 else np.ones(len(dyv), bool)
            sx = float(np.std(dxv[mdx])) if mdx.sum() >= 2 else 0.0
            sy = float(np.std(dyv[mdy])) if mdy.sum() >= 2 else 0.0
            pos_tol_x = pos_nsig * sx
            pos_tol_y = pos_nsig * sy
            print(f'  Stage-1 position scatter: sx={sx:.3f} sy={sy:.3f} px '
                  f'-> position tolerance +/-({pos_tol_x:.3f},{pos_tol_y:.3f}) px '
                  f'({pos_nsig:g}-sigma)')
        gaia_deep = gaia_all.dropna(subset=['RPmag']).copy()
        rp_ab_deep = gaia_deep.RPmag.values + GAIA_RP_AB_OFFSET
        gaia_deep = gaia_deep[rp_ab_deep <= scene_maglim].copy()
        zp2, ze2, sc2, df2, bins2 = _refine_zeropoint_scene(
            image, wcs, gaia_deep, cut_corner, cam, ccd, sector, prf_dir,
            stamp_size, mag_lo, mag_hi, edge_margin, n_jobs,
            zp_ab0=zp_ab, zp_err0=zp_err,
            pos_tol_x=pos_tol_x, pos_tol_y=pos_tol_y,
            var_tol_mag=var_tol_mag, refine_iter=refine_iter, refine_tol=refine_tol,
            max_err_factor=max_err_factor, max_zp_err=max_zp_err,
            zp_bin_width=zp_bin_width, zp_bin_min_n=zp_bin_min_n)
        if zp2 is not None:
            zp_ab, zp_err, zp_scatter, df = zp2, ze2, sc2, df2
            scene_bins = bins2
            stage2_done = True
            print(f'  Adopted scene ZP: {zp_ab:.4f} +/- {zp_err:.4f} mag')
        else:
            print('  Scene refinement unavailable; keeping stage-1 ZP.')

    n_stars_final = int(np.isfinite(df.zp_ab.values).sum())
    if savepath is not None:
        summary = pd.DataFrame({'zp_ab': [zp_ab], 'e_zp_ab': [zp_err],
                                'zp_scatter': [zp_scatter],
                                'n_stars': [n_stars_final]})
        summary_path = f'{savepath}/psf_calibration_zp.csv'
        summary.to_csv(summary_path, index=False)
        print(f'  Zeropoint saved:  {summary_path}')

        if scene_bins:
            bins_df = pd.DataFrame(scene_bins)[['rp_centre', 'median', 'scatter', 'n']]
            bins_path = f'{savepath}/psf_calibration_zp_bins.csv'
            bins_df.to_csv(bins_path, index=False)
            print(f'  ZP mag bins saved: {bins_path}')

        stars = df[['x_pix', 'y_pix', 'ra', 'dec',
                    'flux_psf', 'e_flux_psf', 'background', 'rp_ab']].copy()
        stars.columns = ['x', 'y', 'ra', 'dec', 'flux', 'e_flux', 'background', 'Rp_ab']
        stars_path = f'{savepath}/psf_calibration_stars.csv'
        stars.to_csv(stars_path, index=False)
        print(f'  Star catalog saved: {stars_path}')

    if plot:
        print('  Generating diagnostic figures ...')
        plot_jobs = [
            (_summary_figure,         (image, df, zp_ab, zp_err, savepath)),
            (_colour_magnitude_figure, (df, zp_ab, zp_err, savepath)),
            (_star_fits_pdf,           (df, savepath)),
        ]
        if stage2_done:
            plot_jobs.append((
                _stage_comparison_figure,
                (df_stage1, zp_stage1, ze_stage1, df, zp_ab, zp_err, zp_scatter,
                 scene_bins, savepath),
            ))
            plot_jobs.append((
                _shift_figure,
                (df_stage1, df, pos_tol_x, pos_tol_y, savepath),
            ))
        for fn, args in plot_jobs:
            try:
                _t = time.time()
                fn(*args)
                print(f'    {fn.__name__}: {time.time() - _t:.1f}s', flush=True)
            except Exception:
                import traceback
                print(f'  WARNING: {fn.__name__} failed:')
                traceback.print_exc()

    print(f'=== Calibration complete: ZP={zp_ab:.4f} +/- {zp_err:.4f} '
          f'[total {time.time() - _t_start:.1f}s] ===')
    return zp_ab, zp_err, df


# ---------------------------------------------------------------------------
# Per-frame detection limits
# ---------------------------------------------------------------------------

def _default_time_bins(sector):
    """Return default time bin strings for a given sector cadence."""
    if sector < 27:
        return ['30min', '1hr', '2hr', '6hr', '24hr']
    elif sector < 56:
        return ['10min', '30min', '1hr', '3hr', '12hr', '24hr']
    else:
        return ['200sec', '30min', '1hr', '6hr', '24hr']


def _parse_time_bins(time_bins, time_array):
    """Convert time bin strings to frame counts using the actual time array cadence."""
    import re
    conversions = {'sec': 1/86400, 'min': 1/1440, 'hr': 1/24, 'day': 1}
    cadence_days = float(np.nanmedian(np.diff(time_array)))
    frame_bins = []
    labels = []
    for b in time_bins:
        v, u = re.match(r'([\d.]+)(\w+)', b.strip()).groups()
        resolution_days = float(v) * conversions[u]
        n = max(1, int(round(resolution_days / cadence_days)))
        frame_bins.append(n)
        labels.append(b.strip())
    return frame_bins, labels


def _bin_flux(flux, n):
    """Average flux in blocks of n frames. Returns shape (n_bins, ny, nx)."""
    n_bins = flux.shape[0] // n
    return flux[:n_bins * n].reshape(n_bins, n, flux.shape[1], flux.shape[2]).mean(axis=1)


def _frame_noise(flux_3d):
    """Spatial MAD per frame → per-pixel sigma. Returns shape (n_frames,)."""
    sigma = np.zeros(flux_3d.shape[0])
    for i in range(flux_3d.shape[0]):
        frame = flux_3d[i]
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            sigma[i] = np.nan
        else:
            med = np.median(finite)
            sigma[i] = np.median(np.abs(finite - med)) * 1.4826
    return sigma


def compute_detection_limits(reduced_flux, time_array, zp_ab,
                              sector, cam, ccd,
                              cut_corner=(0, 0),
                              prf_path=PRF_PATH_DEFAULT,
                              stamp_size=9,
                              time_bins=None,
                              savepath=None):
    """
    Compute PSF detection limits for one or more time bin sizes.

    For each bin size N, N consecutive frames are averaged and the background
    noise is measured from the actual binned data via spatial MAD.  The PSF
    matched-filter noise factor converts per-pixel noise to point-source flux
    uncertainty.

    Parameters
    ----------
    reduced_flux : ndarray, shape (n_frames, ny, nx)
        Background-subtracted flux cube (e.g. ReducedFlux.npy).
    time_array : ndarray, shape (n_frames,)
        MJD timestamps for each frame (e.g. Times.npy).
    zp_ab : float
        AB zeropoint from run_calibration.
    sector, cam, ccd : int
    cut_corner : (x0, y0)
        CCD pixel of the cut's bottom-left corner (from find_cuts).
    prf_path : str
    stamp_size : int
    time_bins : list of str or None
        Time bin sizes as strings, e.g. ['10min', '30min', '1hr', '12hr'].
        Uses the same format as the tessellate transient search.
        None uses cadence-appropriate defaults.
    savepath : str or None
        If given, saves ``detection_limits.csv`` here.

    Returns
    -------
    results : dict  keyed by n_bin (int)
        Each value is a dict with keys sigma_bg, mag_lim_3sigma,
        mag_lim_5sigma, mag_lim_10sigma — all 1D arrays over binned frames.
    """
    reduced_flux = np.asarray(reduced_flux, dtype=float)
    time_array = np.asarray(time_array, dtype=float)
    n_frames, ny, nx = reduced_flux.shape

    if time_bins is None:
        time_bins = _default_time_bins(sector)

    frame_bins, labels = _parse_time_bins(time_bins, time_array)

    # PRF at cut centre — matched-filter noise factor
    x0, y0 = cut_corner
    ccd_x = x0 + nx // 2
    ccd_y = y0 + ny // 2

    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)

    cent = stamp_size // 2
    p = prf.locate(float(cent), float(cent), (stamp_size, stamp_size))
    p = p / np.nansum(p)
    prf_noise_factor = 1.0 / np.sqrt(np.nansum(p**2))

    def _limits(sigma_bg, n_bins):
        out = {}
        for nsig, col in [(3, 'mag_lim_3sigma'), (5, 'mag_lim_5sigma'), (10, 'mag_lim_10sigma')]:
            flux = nsig * sigma_bg * prf_noise_factor
            mag = np.full(n_bins, np.nan)
            ok = flux > 0
            mag[ok] = zp_ab - 2.5 * np.log10(flux[ok])
            out[col] = mag
        return out

    results = {}

    print(f'\nDetection limits ({n_frames} frames):')

    for label, n_bin in zip(labels, frame_bins):
        _t = time.time()
        binned = _bin_flux(reduced_flux, n_bin)
        n_bins = binned.shape[0]
        sigma_bg = _frame_noise(binned)
        lims = _limits(sigma_bg, n_bins)

        results[label] = {'n_frames_binned': n_bin, 'sigma_bg': sigma_bg, **lims}

        med5 = np.nanmedian(lims['mag_lim_5sigma'])
        print(f'  {label:>8s}  ({n_bin} frames, {n_bins} bins)  ->  '
              f'5-sigma median: {med5:.3f} AB mag  [{time.time() - _t:.1f}s]')

        if savepath is not None:
            df_lim = pd.DataFrame({
                'bin_index': np.arange(n_bins),
                'sigma_bg': sigma_bg,
                'mag_lim_3sigma': lims['mag_lim_3sigma'],
                'mag_lim_5sigma': lims['mag_lim_5sigma'],
                'mag_lim_10sigma': lims['mag_lim_10sigma'],
            })
            fout = f'{savepath}/detection_limits_{label}.csv'
            df_lim.to_csv(fout, index=False)
            print(f'  Saved: {fout}')

    return results


# ---------------------------------------------------------------------------
# Forced PSF photometry light curves (differenced images)
# ---------------------------------------------------------------------------

# AB reference flux density: mag_AB = -2.5*log10(f / 3631 Jy)
_AB_ZERO_JY = 3631.0
_UNIT_SCALE = {'jy': 1.0, 'mjy': 1e3, 'mujy': 1e6, 'ujy': 1e6, 'µjy': 1e6,
               'μjy': 1e6}


def _convert_flux_units(flux, e_flux, units, zp_ab):
    """
    Convert counts (e/s) to the requested units.

    'count'/'counts' -> raw e/s (no zeropoint needed).
    'Jy'/'mJy'/'muJy' -> flux density, linear in flux (negative values kept,
        as expected for difference photometry).
    'mag' -> AB magnitude; negative/zero flux is a non-detection (NaN).
    """
    u = units.lower()
    if u in ('count', 'counts'):
        return flux, e_flux, 'counts'
    if zp_ab is None:
        raise ValueError(f"units='{units}' requires a zeropoint (zp_ab).")

    if u == 'mag':
        flux = np.asarray(flux, dtype=float)
        e_flux = np.asarray(e_flux, dtype=float)
        mag = np.full_like(flux, np.nan)
        emag = np.full_like(flux, np.nan)

        pos = flux > e_flux
        mag[pos] = zp_ab - 2.5 * np.log10(flux[pos])
        mag[~pos] =  zp_ab - 2.5 * np.log10(e_flux[~pos])
        # Exact magnitude offset to the +1 sigma (brighter) flux; finite and
        # correct for faint points, unlike the linear 1.0857*e/flux which
        # diverges as e_flux -> flux.  (~1.0857*e/flux at high S/N.)
        emag[pos] = 2.5 * np.log10(1.0 + e_flux[pos] / flux[pos])



        return mag, emag, 'mag'

    scale = _UNIT_SCALE.get(u)
    if scale is None:
        raise ValueError(f"Unknown units '{units}'. Use counts, Jy, mJy, muJy, or mag.")
    # f_Jy = 3631 * 10^(-0.4*zp_ab) * flux_counts  (linear, so signs preserved)
    c = _AB_ZERO_JY * 10 ** (-0.4 * zp_ab) * scale
    return flux * c, e_flux * c, units


def psf_lightcurve(flux_cube, time_array, x, y, sector, cam, ccd,
                   cut_corner=(0, 0), prf_path=PRF_PATH_DEFAULT,
                   stamp_size=9, neighbours=None, units='counts', zp_ab=None):
    """
    Forced PSF photometry light curve at a FIXED position on a (differenced)
    flux cube, using the same fixed-position scene model as the calibration.

    The TESS PRF is evaluated once at the source's fixed sub-pixel position;
    because the position never changes, the design matrix [PRF, (neighbours,)
    flat background] is built once and applied to every frame as a single linear
    projection (a matched filter).  This gives an optimal, background-limited
    flux(t) and e_flux(t) in one vectorised solve over the whole cube.

    Parameters
    ----------
    flux_cube : (n_frames, ny, nx) ndarray
        The differenced/reduced flux cube for the cut.
    time_array : (n_frames,) ndarray
    x, y : float
        Source position in CUT-frame pixels (e.g. xcentroid_psf, ycentroid_psf
        from the event table).  Held fixed.
    sector, cam, ccd : int
    cut_corner : (x0, y0)
        CCD pixel coordinate of the cut's bottom-left corner (for the PRF).
    prf_path : str
        Root directory of local TESS PRF files.
    stamp_size : int (odd)
        Stamp width used for the fit.
    neighbours : (M, 2) array or None
        Nearby source positions (cut-frame pixels) to deblend as extra fixed
        PRF components.  On difference images static neighbours subtract out, so
        this is usually only needed for nearby *variable* sources.
    units : str
        Output units: 'counts' (default, raw e/s), 'Jy', 'mJy', 'muJy', or
        'mag'.  All except 'counts' require zp_ab.  Flux-density units are
        linear in flux (negative difference-flux preserved); 'mag' treats
        negative/zero flux as a non-detection (NaN).
    zp_ab : float or None
        AB zeropoint (from the cut's calibration), required for non-count units.

    Returns
    -------
    dict with keys: time, flux, e_flux (in `units`), units, and the raw
    flux_counts, e_flux_counts, background (all arrays over frames).
    """
    flux_cube = np.asarray(flux_cube, dtype=float)
    nfr, ny, nx = flux_cube.shape
    half = stamp_size // 2

    xi, yi = int(np.round(x)), int(np.round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > nx or yi + half + 1 > ny:
        raise ValueError(f'Source ({x:.1f},{y:.1f}) too close to the cut edge '
                         f'for a {stamp_size}px stamp.')

    stamp_cube = flux_cube[:, yi - half:yi + half + 1, xi - half:xi + half + 1]

    neigh_sub = None
    if neighbours is not None:
        neigh_sub = [(nxp - xi, nyp - yi) for nxp, nyp in np.asarray(neighbours, float)
                     if (abs(nxp - x) <= half and abs(nyp - y) <= half
                         and not (nxp == x and nyp == y))]

    return _psf_lc_core(stamp_cube, time_array, x - xi, y - yi,
                        cut_corner[0] + x, cut_corner[1] + y, cam, ccd, sector,
                        prf_path=prf_path, stamp_size=stamp_size,
                        neighbours_sub=neigh_sub, units=units, zp_ab=zp_ab)


def _psf_lc_core(stamp_cube, time_array, x_sub, y_sub, ccd_x, ccd_y, cam, ccd,
                 sector, prf_path=PRF_PATH_DEFAULT, stamp_size=9,
                 neighbours_sub=None, units='counts', zp_ab=None):
    """
    PSF-photometry light curve from a PRE-EXTRACTED stamp cube.

    stamp_cube : (n_frames, stamp_size, stamp_size), already centred on the
        source's nearest pixel.  x_sub, y_sub : sub-pixel offset of the source
        from the stamp centre.  ccd_x, ccd_y : CCD position (for the PRF).
        neighbours_sub : list of (dx, dy) neighbour offsets from the stamp centre.

    Splitting extraction from the solve lets callers slice the stamp from an
    in-RAM cube and pass only the small stamp to workers (no big-cube memmap).
    """
    stamp_cube = np.asarray(stamp_cube, dtype=float)
    nfr = stamp_cube.shape[0]
    cent = stamp_size // 2
    npix = stamp_size * stamp_size

    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)

    def _col(dx, dy):
        p = prf.locate(cent + dx, cent + dy, (stamp_size, stamp_size))
        s = np.nansum(p)
        return (p / s).ravel() if (np.isfinite(s) and s > 0) else np.zeros(npix)

    cols = [_col(x_sub, y_sub)]
    if neighbours_sub is not None:
        for dx, dy in neighbours_sub:
            cols.append(_col(dx, dy))
    cols.append(np.ones(npix))           # flat background
    A = np.column_stack(cols)            # npix x M  (target, neighbours..., bg)

    D = stamp_cube.reshape(nfr, npix)
    good = np.all(np.isfinite(D), axis=0) & np.all(np.isfinite(A), axis=1)
    if good.sum() < A.shape[1] + 1:
        raise RuntimeError('Too few finite pixels for PSF photometry.')
    Ag = A[good]
    Dg = D[:, good]

    ATA_inv = np.linalg.inv(Ag.T @ Ag)
    coeffs = (ATA_inv @ Ag.T) @ Dg.T     # M x n_frames
    flux = coeffs[0]
    bg = coeffs[-1]

    resid = Dg.T - Ag @ coeffs
    med = np.median(resid, axis=0)
    sigma = 1.4826 * np.median(np.abs(resid - med), axis=0)
    e_flux = sigma * np.sqrt(ATA_inv[0, 0])

    value, e_value, unit_label = _convert_flux_units(flux, e_flux, units, zp_ab)
    return {'time': np.asarray(time_array, dtype=float),
            'flux': value, 'e_flux': e_value, 'units': unit_label,
            'flux_counts': flux, 'e_flux_counts': e_flux, 'background': bg}


def _aperture_lc_core(stamp_cube, time_array, x_sub, y_sub, ccd_x, ccd_y, cam, ccd,
                      sector, prf_path=PRF_PATH_DEFAULT, stamp_size=9, radius=1.5,
                      units='counts', zp_ab=None):
    """
    Aperture-photometry light curve from a PRE-EXTRACTED stamp cube.

    Sums the central (2*floor(radius)+1) box per frame, with a background-limited
    error from the stamp scatter.  For non-count units the zeropoint is aperture-
    corrected for the PRF flux lost outside the box.
    """
    stamp_cube = np.asarray(stamp_cube, dtype=float)
    cent = stamp_size // 2
    buf = int(np.floor(radius))
    box = stamp_cube[:, cent - buf:cent + buf + 1, cent - buf:cent + buf + 1]
    flux = np.nansum(box, axis=(1, 2))

    med = np.nanmedian(stamp_cube, axis=(1, 2), keepdims=True)
    sig = 1.4826 * np.nanmedian(np.abs(stamp_cube - med), axis=(1, 2))
    e_flux = np.sqrt((2 * buf + 1) ** 2) * sig

    unit_label = 'counts'
    value, e_value = flux, e_flux
    if units.lower() not in ('count', 'counts'):
        frac = aperture_correction(sector, cam, ccd, ccd_x, ccd_y, x_sub, y_sub,
                                   radius=radius, prf_path=prf_path)
        zp_eff = zp_ab + 2.5 * np.log10(frac)   # aperture loss folded into the ZP
        value, e_value, unit_label = _convert_flux_units(flux, e_flux, units, zp_eff)

    return {'time': np.asarray(time_array, dtype=float),
            'flux': value, 'e_flux': e_value, 'units': unit_label,
            'flux_counts': flux, 'e_flux_counts': e_flux,
            'background': np.squeeze(med)}


def aperture_correction(sector, cam, ccd, ccd_x, ccd_y, x_sub=0.0, y_sub=0.0,
                        radius=1.5, prf_path=PRF_PATH_DEFAULT, stamp_size=13):
    """
    Fraction of a point source's PRF flux captured by Generate_LC's 'sum'
    aperture -- a (2*floor(radius)+1) box centred on the nearest pixel.

    The PSF zeropoint assumes total flux, but aperture photometry only sums the
    box, so the aperture-corrected zeropoint is

        zp_aperture = zp_ab + 2.5 * log10(fraction).

    x_sub, y_sub : sub-pixel offset of the source from the box-centre pixel.
    """
    prf_dir = f'{prf_path}/Sectors4+' if sector >= 4 else f'{prf_path}/Sectors1_2_3'
    prf = _get_prf(cam, ccd, sector, ccd_x, ccd_y, prf_dir)
    cent = stamp_size // 2
    p = prf.locate(cent + x_sub, cent + y_sub, (stamp_size, stamp_size))
    p = p / np.nansum(p)                       # normalise to total flux
    buf = int(np.floor(radius))
    box = p[cent - buf:cent + buf + 1, cent - buf:cent + buf + 1]
    return float(np.nansum(box))


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _sigma_clip_mask(values, nsigma=3, maxiter=5):
    """Return boolean mask of inliers after iterative sigma clipping."""
    mask = np.isfinite(values)
    for _ in range(maxiter):
        med = np.median(values[mask])
        std = np.std(values[mask])
        new_mask = np.isfinite(values) & (np.abs(values - med) < nsigma * std)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return mask


def _err_keep_mask(e_zp, max_err_factor, max_zp_err=None):
    """
    Boolean mask of points to keep: finite, positive error, error not larger than
    max_err_factor times the median error (drops poorly-constrained fits), and
    (if set) error not larger than the absolute cap max_zp_err.
    """
    e = np.asarray(e_zp, dtype=float)
    keep = np.isfinite(e) & (e > 0)
    if keep.sum() >= 2 and np.isfinite(max_err_factor) and max_err_factor > 0:
        emed = float(np.median(e[keep]))
        keep &= e <= max_err_factor * emed
    if max_zp_err is not None and np.isfinite(max_zp_err) and max_zp_err > 0:
        keep &= e <= max_zp_err
    return keep


def _binned_zeropoint(zp_vals, rp_ab, bin_width=0.5, min_bin_n=5):
    """
    Empirical magnitude-binned robust zeropoint combine.

    The per-source ZP scatter is heteroscedastic (grows toward faint mags), so a
    single mean/clip is dragged by the faint wing.  Instead, bin by Rp_AB, take a
    robust centre (median) and scatter (1.4826*MAD) per bin, and combine the bin
    medians inverse-variance weighted by the EMPIRICAL per-bin median error.
    Faint, noisy bins down-weight themselves without being cut.

    Returns (zp_ab, zp_err, zp_scatter, bins) where bins is a list of dicts
    (rp_centre, median, scatter, n).  Falls back to a global median/MAD if fewer
    than two usable bins.

    zp is a log (magnitude) quantity, so every centre/combine below (the
    per-bin median, the global fallback, and the final inverse-variance
    combine across bins) is done in linear flux space via
    _zp_to_scale/_scale_to_zp and converted back -- not a raw median/weighted
    mean of the magnitudes.  Only the scatter (a *width*, not a location) is
    reported in magnitude units, per the usual photometric convention.
    """
    zp = np.asarray(zp_vals, dtype=float)
    rp = np.asarray(rp_ab, dtype=float)
    ok = np.isfinite(zp) & np.isfinite(rp)
    zp, rp = zp[ok], rp[ok]

    def _global():
        med = float(_scale_to_zp(np.median(_zp_to_scale(zp))))
        s = float(1.4826 * np.median(np.abs(zp - med)))
        se = s / np.sqrt(max(len(zp), 1))
        return med, (se if se > 0 else s), s, []

    if len(zp) < min_bin_n:
        return _global()

    lo = np.floor(rp.min() / bin_width) * bin_width
    hi = np.ceil(rp.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + 0.5 * bin_width, bin_width)

    bins = []
    for i in range(len(edges) - 1):
        sel = (rp >= edges[i]) & (rp < edges[i + 1])
        n = int(sel.sum())
        if n < min_bin_n:
            continue
        v = zp[sel]
        med = float(_scale_to_zp(np.median(_zp_to_scale(v))))
        s = float(1.4826 * np.median(np.abs(v - med)))
        if not (s > 0):
            s = float(np.std(v))
        if not (s > 0):
            continue
        bins.append({'rp_centre': float(0.5 * (edges[i] + edges[i + 1])),
                     'median': med, 'scatter': s, 'n': n,
                     'se': s / np.sqrt(n)})

    if len(bins) < 2:
        return _global()

    meds = np.array([b['median'] for b in bins])
    ses = np.array([b['se'] for b in bins])
    scat = np.array([b['scatter'] for b in bins])
    w = 1.0 / ses**2
    zp_ab = float(_scale_to_zp(np.sum(w * _zp_to_scale(meds)) / np.sum(w)))
    zp_err = float(np.sqrt(1.0 / np.sum(w)))
    # Core width: inverse-variance-weighted mean of the per-bin robust scatter
    zp_scatter = float(np.sum(w * scat) / np.sum(w))
    return zp_ab, zp_err, zp_scatter, bins


def _summary_figure(image, df, zp_ab, zp_err, savepath):
    import matplotlib.gridspec as gridspec

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    zp_vals = df.zp_ab.values
    e_zp = df.e_zp_ab.values
    x_pix = df.x_pix.values
    y_pix = df.y_pix.values

    inliers = _sigma_clip_mask(zp_vals)
    zp_in = zp_vals[inliers]
    med_zp = float(np.median(zp_in))
    std_zp = float(np.std(zp_in))
    n_in = int(inliers.sum())

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.32)

    # ---- Panel 1: ZP histogram ------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bins = int(np.clip(np.sqrt(n_in), 10, 40))
    ax1.hist(zp_in, bins=bins, color='C0', alpha=0.75, edgecolor='white', linewidth=0.5)
    ax1.axvline(zp_ab,  color='k',   ls='--', lw=1.5, label=f'Clipped mean = {zp_ab:.3f}')
    ax1.axvline(med_zp, color='C3',  ls=':',  lw=1.5, label=f'Median = {med_zp:.3f}')
    ax1.axvspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.15, color='k',
                label=f'$\\pm$1$\\sigma$ = {zp_err:.4f}')
    stats_txt = (f'N = {n_in}  (of {len(df)} fitted)\n'
                 f'std = {std_zp:.4f} mag')
    ax1.text(0.97, 0.96, stats_txt, transform=ax1.transAxes,
             ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax1.set_xlabel('Per-star ZP (AB mag)')
    ax1.set_ylabel('N stars')
    ax1.set_title('Zeropoint distribution (3$\\sigma$ clipped)')
    ax1.legend(fontsize=7, loc='upper left')

    # ---- Panel 2: Star locations on reference image --------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    nonan = np.isfinite(image)
    vlo = float(np.percentile(image[nonan], 2))
    vhi = float(np.percentile(image[nonan], 98))
    im2 = ax2.imshow(image, origin='lower', cmap='gray', vmin=vlo, vmax=vhi, aspect='auto')
    sc2 = ax2.scatter(x_pix[inliers],  y_pix[inliers],
                      c=zp_vals[inliers], cmap='RdYlGn_r',
                      vmin=med_zp - 3*std_zp, vmax=med_zp + 3*std_zp,
                      s=60, edgecolors='white', linewidths=0.5, zorder=3,
                      label='Inliers')
    ax2.scatter(x_pix[~inliers], y_pix[~inliers],
                marker='x', s=40, color='C3', linewidths=1.0, zorder=4,
                label='Outliers')
    plt.colorbar(sc2, ax=ax2, label='ZP (AB mag)')
    ax2.set_xlabel('Column (px)')
    ax2.set_ylabel('Row (px)')
    ax2.set_title('Calibration star locations')
    ax2.legend(fontsize=7, loc='upper right')

    # ---- Panel 3: ZP vs x_pix ------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(x_pix[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7, elinewidth=0.8,
                 capsize=2)
    if n_in >= 2:
        cx = np.polyfit(x_pix[inliers], zp_vals[inliers], 1)
        xg = np.linspace(x_pix[inliers].min(), x_pix[inliers].max(), 100)
        ax3.plot(xg, np.polyval(cx, xg), color='C3', lw=1.2,
                 label=f'slope={cx[0]*1e3:.2f}e-3 mag/px')
        ax3.legend(fontsize=7)
    ax3.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.6)
    ax3.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k')
    ax3.set_xlabel('x pixel (cut frame)')
    ax3.set_ylabel('ZP (AB mag)')
    ax3.set_title('Zeropoint vs detector column')

    # ---- Panel 4: ZP vs y_pix ------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar(y_pix[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7, elinewidth=0.8,
                 capsize=2)
    if n_in >= 2:
        cy = np.polyfit(y_pix[inliers], zp_vals[inliers], 1)
        yg = np.linspace(y_pix[inliers].min(), y_pix[inliers].max(), 100)
        ax4.plot(yg, np.polyval(cy, yg), color='C3', lw=1.2,
                 label=f'slope={cy[0]*1e3:.2f}e-3 mag/px')
        ax4.legend(fontsize=7)
    ax4.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.6)
    ax4.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k')
    ax4.set_xlabel('y pixel (cut frame)')
    ax4.set_ylabel('ZP (AB mag)')
    ax4.set_title('Zeropoint vs detector row')

    fig.suptitle(f'PSF Flux Calibration  —  ZP = {zp_ab:.4f} ± {zp_err:.4f} AB mag  '
                 f'(N={n_in})', fontsize=11, y=1.01)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_diagnostic.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Summary diagnostic: {fname}')
    else:
        plt.show()
    plt.close(fig)


def _colour_magnitude_figure(df, zp_ab, zp_err, savepath):

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    inliers = _sigma_clip_mask(df.zp_ab.values)
    rp_ab = df.rp_ab.values
    zp_vals = df.zp_ab.values
    e_zp = df.e_zp_ab.values
    bp_rp = df.bp_rp.values if 'bp_rp' in df.columns else np.full(len(df), np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)

    # ---- Top: source magnitude vs ZP -----------------------------------------
    ax1.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.7, zorder=1, label=f'ZP={zp_ab:.4f}')
    ax1.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k', zorder=1)
    ax1.errorbar(rp_ab[inliers], zp_vals[inliers], yerr=e_zp[inliers],
                 fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7,
                 elinewidth=0.8, capsize=2, zorder=3, label='Inliers')
    if inliers.sum() >= 2:
        cm = np.polyfit(rp_ab[inliers], zp_vals[inliers], 1)
        xg = np.linspace(rp_ab[inliers].min(), rp_ab[inliers].max(), 100)
        ax1.plot(xg, np.polyval(cm, xg), color='C3', lw=1.2, zorder=2,
                 label=f'slope={cm[0]:.4f} mag/mag')
    ax1.set_xlabel('Gaia Rp (AB mag)')
    ax1.set_ylabel('Per-star ZP (AB mag)')
    ax1.set_title('Zeropoint vs source magnitude')
    ax1.legend(fontsize=7)

    # ---- Bottom: ZP vs BP-RP colour ------------------------------------------
    has_colour = np.isfinite(bp_rp).any()
    if has_colour:
        ok = inliers & np.isfinite(bp_rp)
        ax2.axhline(zp_ab, color='k', ls='--', lw=1.0, alpha=0.7, zorder=1)
        ax2.axhspan(zp_ab - zp_err, zp_ab + zp_err, alpha=0.12, color='k', zorder=1)
        ax2.errorbar(bp_rp[ok], zp_vals[ok], yerr=e_zp[ok],
                     fmt='o', ms=4, color='C0', ecolor='C0', alpha=0.7,
                     elinewidth=0.8, capsize=2, zorder=3, label='Inliers')
        if ok.sum() >= 2:
            cc = np.polyfit(bp_rp[ok], zp_vals[ok], 1)
            cg = np.linspace(bp_rp[ok].min(), bp_rp[ok].max(), 100)
            ax2.plot(cg, np.polyval(cc, cg), color='C3', lw=1.2, zorder=2,
                     label=f'slope={cc[0]:.4f} mag/mag')
        ax2.set_xlabel('Gaia BP - RP (mag)')
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'BP-RP colour not available in catalog',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=9)
    ax2.set_ylabel('Per-star ZP (AB mag)')
    ax2.set_title('Zeropoint vs Gaia colour')

    fig.suptitle(f'PSF Calibration Colour Diagnostics  —  ZP = {zp_ab:.4f} ± {zp_err:.4f}',
                 fontsize=10)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_colour.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Colour diagnostic:  {fname}')
    else:
        plt.show()
    plt.close(fig)


def _stage_comparison_figure(df1, zp1, ze1, df2, zp2, se2, sc2, bins, savepath):
    """
    Compare the stage-1 (isolated) and stage-2 (scene) zeropoint solutions.
    Bands show the core scatter (ze1 for stage 1; sc2 for stage 2); se2 is the
    stage-2 combine precision (shown in the text box).
    """

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    # Drop massive outliers for display only (does not affect the ZP, which is
    # already robustly estimated).  Keep points within a generous robust window.
    def _disp_mask(z):
        z = np.asarray(z, dtype=float)
        med = np.nanmedian(z)
        mad = 1.4826 * np.nanmedian(np.abs(z - med))
        w = max(6.0 * mad, 0.75)
        return np.isfinite(z) & (np.abs(z - med) < w)

    df1 = df1[_disp_mask(df1.zp_ab.values)]
    df2 = df2[_disp_mask(df2.zp_ab.values)]

    z1 = df1.zp_ab.values
    z2 = df2.zp_ab.values
    rp1 = df1.rp_ab.values
    rp2 = df2.rp_ab.values

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

    # ---- Panel 1: per-star ZP histograms + global values ---------------------
    lo = np.nanpercentile(np.concatenate([z1, z2]), 1)
    hi = np.nanpercentile(np.concatenate([z1, z2]), 99)
    hist_bins = np.linspace(lo, hi, 30)
    ax1.hist(z1, bins=hist_bins, color='C0', alpha=0.55, label=f'Stage 1 (N={len(z1)})')
    ax1.hist(z2, bins=hist_bins, color='C1', alpha=0.55, label=f'Stage 2 (N={len(z2)})')
    ax1.axvline(zp1, color='C0', ls='--', lw=1.5)
    ax1.axvspan(zp1 - ze1, zp1 + ze1, color='C0', alpha=0.12)
    ax1.axvline(zp2, color='C1', ls='--', lw=1.5)
    ax1.axvspan(zp2 - sc2, zp2 + sc2, color='C1', alpha=0.12)
    txt = (f'Stage 1: {zp1:.4f} (scatter {ze1:.4f})\n'
           f'Stage 2: {zp2:.4f} (scatter {sc2:.4f}, SE {se2:.4f})\n'
           f'$\\Delta$ZP = {zp2 - zp1:+.4f} mag')
    ax1.text(0.97, 0.97, txt, transform=ax1.transAxes, ha='right', va='top',
             fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax1.set_xlabel('Per-star ZP (AB mag)')
    ax1.set_ylabel('N stars')
    ax1.set_title('Zeropoint distributions')
    ax1.legend(fontsize=7, loc='upper left')

    # ---- Panel 2: per-star ZP vs magnitude, both stages ----------------------
    ax2.axhline(zp1, color='C0', ls='--', lw=1.0, alpha=0.7)
    ax2.axhspan(zp1 - ze1, zp1 + ze1, color='C0', alpha=0.10)
    ax2.axhline(zp2, color='C1', ls='--', lw=1.0, alpha=0.7)
    ax2.axhspan(zp2 - sc2, zp2 + sc2, color='C1', alpha=0.10)
    ax2.scatter(rp1, z1, s=18, color='C0', alpha=0.5, label='Stage 1')
    ax2.scatter(rp2, z2, s=18, color='C1', alpha=0.4, marker='s', label='Stage 2')
    # Empirical per-magnitude-bin median +/- robust scatter (the combine weights)
    if bins:
        bc = np.array([b['rp_centre'] for b in bins])
        bm = np.array([b['median'] for b in bins])
        bs = np.array([b['scatter'] for b in bins])
        ax2.errorbar(bc, bm, yerr=bs, fmt='o-', color='k', ms=5, lw=1.4,
                     capsize=3, zorder=6, label='Stage 2 mag bins (median$\\pm$MAD)')
    ax2.set_xlabel('Gaia Rp (AB mag)')
    ax2.set_ylabel('Per-star ZP (AB mag)')
    ax2.set_title('Zeropoint vs magnitude')
    ax2.legend(fontsize=7)

    # ---- Panel 3: matched per-star comparison (same stars, both stages) ------
    # Match by sky position (nearest within 1 arcsec)
    ra1, dec1 = df1.ra.values, df1.dec.values
    ra2, dec2 = df2.ra.values, df2.dec.values
    mz1, mz2 = [], []
    for i in range(len(df1)):
        cosd = np.cos(np.radians(dec1[i]))
        sep = np.hypot((ra2 - ra1[i]) * cosd, dec2 - dec1[i]) * 3600.0
        j = int(np.argmin(sep))
        if sep[j] < 1.0:
            mz1.append(z1[i])
            mz2.append(z2[j])
    mz1 = np.array(mz1)
    mz2 = np.array(mz2)
    if len(mz1) >= 1:
        ax3.scatter(mz1, mz2, s=20, color='C3', alpha=0.7)
        lim_lo = float(np.nanmin([mz1.min(), mz2.min()]))
        lim_hi = float(np.nanmax([mz1.max(), mz2.max()]))
        ax3.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1.0, alpha=0.6,
                 label='1:1')
        dmed = float(np.median(mz2 - mz1))
        ax3.text(0.03, 0.97,
                 f'matched N={len(mz1)}\nmedian $\\Delta$={dmed:+.4f} mag',
                 transform=ax3.transAxes, ha='left', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax3.legend(fontsize=7, loc='lower right')
    else:
        ax3.text(0.5, 0.5, 'No stars matched between stages',
                 transform=ax3.transAxes, ha='center', va='center')
    ax3.set_xlabel('Stage 1 per-star ZP (AB mag)')
    ax3.set_ylabel('Stage 2 per-star ZP (AB mag)')
    ax3.set_title('Matched per-star ZP')

    fig.suptitle(f'Stage 1 vs Stage 2 Zeropoint  —  '
                 f'$\\Delta$ZP = {zp2 - zp1:+.4f} mag', fontsize=11)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_stage_comparison.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Stage comparison:   {fname}')
    else:
        plt.show()
    plt.close(fig)


def _shift_figure(df1, df2, pos_tol_x, pos_tol_y, savepath):
    """Per-source fitted position offsets (dx, dy): stage 1 and stage 2."""

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 9,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), tight_layout=True)

    panels = [
        (ax1, df1, 'Stage 1 (isolated)', 'C0', None, None),
        (ax2, df2, 'Stage 2 (scene)', 'C1', pos_tol_x, pos_tol_y),
    ]
    # Common symmetric limits across both panels
    alld = np.concatenate([df1.dx_fit.values, df1.dy_fit.values,
                           df2.dx_fit.values, df2.dy_fit.values])
    alld = alld[np.isfinite(alld)]
    lim = float(np.nanpercentile(np.abs(alld), 99)) * 1.2 if alld.size else 0.5
    lim = max(lim, pos_tol_x, pos_tol_y, 0.05)

    for ax, dfx, title, col, tx, ty in panels:
        dx = dfx.dx_fit.values
        dy = dfx.dy_fit.values
        ok = np.isfinite(dx) & np.isfinite(dy)
        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        ax.scatter(dx[ok], dy[ok], s=18, color=col, alpha=0.7,
                   edgecolors='none')
        # Mean offset marker
        if ok.sum() >= 1:
            mx, my = float(np.mean(dx[ok])), float(np.mean(dy[ok]))
            ax.scatter([mx], [my], s=90, marker='+', color='k', lw=1.8,
                       zorder=5, label=f'mean=({mx:+.3f},{my:+.3f})')
            sx, sy = float(np.std(dx[ok])), float(np.std(dy[ok]))
            ax.text(0.03, 0.97, f'N={ok.sum()}\nstd=({sx:.3f},{sy:.3f}) px',
                    transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        # Position tolerance box (stage 2 only)
        if tx is not None and (tx > 0 or ty > 0):
            ax.add_patch(plt.Rectangle((-tx, -ty), 2 * tx, 2 * ty, fill=False,
                                       ec='C3', ls='--', lw=1.2,
                                       label=f'tol +/-({tx:.3f},{ty:.3f})'))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('dx (px)')
        ax.set_ylabel('dy (px)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('Fitted position offsets', fontsize=11)

    if savepath is not None:
        fname = f'{savepath}/psf_calibration_shifts.pdf'
        fig.savefig(fname, bbox_inches='tight')
        print(f'  Shift diagnostic:   {fname}')
    else:
        plt.show()
    plt.close(fig)


def _star_fits_pdf(df, savepath, max_stars=20):
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 8,
    })

    stars_per_page = 3
    # Cap the number of per-star pages: rendering thousands of multi-panel pages
    # is a slow serial bottleneck.  Keep the brightest sources (lowest Rp_AB).
    if len(df) > max_stars:
        df = df.sort_values('rp_ab').head(max_stars)
        print(f'    (star-fit pages capped to {max_stars} brightest sources)')
    rows = list(df.itertuples())
    n_pages = int(np.ceil(len(rows) / stars_per_page))

    fname = (f'{savepath}/psf_calibration_star_fits.pdf'
             if savepath is not None else '/tmp/psf_calibration_star_fits.pdf')

    with PdfPages(fname) as pdf:
        for page in range(n_pages):
            chunk = rows[page * stars_per_page:(page + 1) * stars_per_page]
            n_rows = len(chunk)
            fig, axes = plt.subplots(n_rows, 3,
                                     figsize=(8.5, 3.0 * n_rows),
                                     squeeze=False)

            for row_i, r in enumerate(chunk):
                stamp = getattr(r, 'stamp_data', None)
                model = getattr(r, 'model_data', None)
                residual = getattr(r, 'residual_data', None)

                if stamp is None:
                    for col_i in range(3):
                        axes[row_i, col_i].axis('off')
                    continue

                # Scale the colour range to the inner 5x5 region used for the fit
                cent = stamp.shape[0] // 2
                inner = 2
                sl = np.s_[cent - inner:cent + inner + 1, cent - inner:cent + inner + 1]
                stamp_inner = stamp[sl]
                res_inner = residual[sl]
                vlo = float(np.nanmin(stamp_inner))
                vhi = float(np.nanmax(stamp_inner))
                res_lim = float(np.nanmax(np.abs(res_inner))) * 1.05

                titles = ['Data', 'PSF model', 'Residual']
                arrays = [stamp, model, residual]
                cmaps = ['gray', 'gray', 'RdBu_r']
                vlims = [(vlo, vhi), (vlo, vhi), (-res_lim, res_lim)]

                star_idx = page * stars_per_page + row_i + 1
                param_line1 = (f'Star {star_idx}  |  '
                               f'RA={r.ra:.5f}  Dec={r.dec:.5f}  '
                               f'(x,y)=({r.x_pix:.1f},{r.y_pix:.1f})')
                param_line2 = (f'flux={r.flux_psf:.1f}±{r.e_flux_psf:.1f}  '
                               f'dx={r.dx_fit:.3f}  dy={r.dy_fit:.3f}  '
                               f'bg={r.background:.2f}  '
                               f'Rp_AB={r.rp_ab:.3f}  '
                               f'ZP={r.zp_ab:.4f}±{r.e_zp_ab:.4f} mag')

                for col_i in range(3):
                    ax = axes[row_i, col_i]
                    im = ax.imshow(arrays[col_i], origin='lower', cmap=cmaps[col_i],
                                   vmin=vlims[col_i][0], vmax=vlims[col_i][1],
                                   interpolation='nearest')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(titles[col_i], fontsize=8)
                    ax.set_xlabel('px')
                    ax.set_ylabel('px')

                # Place parameter text inside the data panel to avoid layout issues
                axes[row_i, 1].set_title(
                    f'{titles[1]}\n{param_line1}\n{param_line2}',
                    fontsize=6, pad=4
                )

            fig.suptitle(f'Per-star PSF fits  (page {page+1}/{n_pages})',
                         fontsize=10, y=1.0)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'  Star fit pages:     {fname}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='PSF flux calibration to Gaia Rp AB zeropoint.')
    parser.add_argument('fits_path',
                        help='FITS file with image in HDU[1] and CCD WCS header.')
    parser.add_argument('--sector',   type=int, required=True)
    parser.add_argument('--cam',      type=int, required=True)
    parser.add_argument('--ccd',      type=int, required=True)
    parser.add_argument('--cut-x0',   type=float, default=0.0,
                        help='CCD x-pixel of cut bottom-left corner (from find_cuts).')
    parser.add_argument('--cut-y0',   type=float, default=0.0,
                        help='CCD y-pixel of cut bottom-left corner (from find_cuts).')
    parser.add_argument('--gaia-path', default=GAIA_PATH_DEFAULT)
    parser.add_argument('--prf-path',  default=PRF_PATH_DEFAULT)
    parser.add_argument('--mag-lo',   type=float, default=12.0)
    parser.add_argument('--mag-hi',   type=float, default=14.0)
    parser.add_argument('--iso-radius', type=float, default=5.0)
    parser.add_argument('--stamp-size', type=int,   default=9)
    parser.add_argument('--n-jobs',   type=int, default=-1)
    parser.add_argument('--plot',     action='store_true')
    parser.add_argument('--savepath', default=None)
    args = parser.parse_args()

    with fits.open(args.fits_path) as f:
        img = f[1].data.astype(float)
        w = WCS(f[1].header)

    zp, zp_e, _ = run_calibration(
        img, w,
        sector=args.sector, cam=args.cam, ccd=args.ccd,
        cut_corner=(args.cut_x0, args.cut_y0),
        gaia_path=args.gaia_path,
        prf_path=args.prf_path,
        mag_lo=args.mag_lo, mag_hi=args.mag_hi,
        iso_radius_pix=args.iso_radius,
        stamp_size=args.stamp_size,
        n_jobs=args.n_jobs,
        plot=args.plot, savepath=args.savepath,
    )
