"""
Moving-object (SSO) mask for the reduction's background.

The reference subtraction removes static sources but never asteroids, so every predicted asteroid
position has to be excluded from every background estimate, or the background absorbs part of the
asteroid's flux (0.2-9 % in S27 cam1 ccd2 cut 50, background_leak/). build_sso_mask turns a cut's
predicted ephemeris (asteroids/<base>_Asteroids.parquet, from predict_asteroids, which runs ahead of
reduce) into a bool (T, Y, X) cube on the reduced frame list, for tessreduce.reduce(moving_mask=).

Each predicted position is masked with a capsule: every pixel within r of the asteroid's path over
the exposure (mid-exposure position +/- half the per-frame motion), r = max(R_MIN, r_iso), where
r_iso is the radius beyond which the PRF-predicted pixel flux drops below ISO_FRAC sigma_pix.
"""

import numpy as np

from .asteroid_photometry import match_ephemeris_to_reduced_frames
from .tools import load_table, table_exists

ZP_600S = 21.21     # provisional, empirical: S27 cut 50 photometry at 600 s; scaled by exposure below
R_MIN = 3.0         # px, smallest mask radius
R_MAX = 6.5         # px, half-width of the 13x13 PRF model -- brighter objects need empirical wings beyond this
ISO_FRAC = 0.25     # mask out to where the PRF-predicted pixel flux drops to this fraction of sigma_pix


def measure_sigma_pix(flux, n_pairs=10):
    """Per-pixel noise of a difference image for this cut, measured from the data: the robust std
    (1.4826 x MAD) of consecutive-frame differences / sqrt(2), over n_pairs pairs spread through the
    sector. Static sources cancel in the frame difference, so this needs neither a reference nor a
    source mask, and is available before reduce() runs (the reference is a stack, so the difference
    image's noise is close to a single frame's)."""
    flux = np.asarray(flux)
    idx = np.linspace(0, len(flux) - 2, n_pairs).astype(int)
    s = []
    for i in idx:
        d = (np.asarray(flux[i + 1], dtype=float) - np.asarray(flux[i], dtype=float)).ravel()
        d = d[np.isfinite(d) & (d != 0)]
        if len(d) > 100:
            s.append(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    return float(np.median(s)) if s else np.nan


def sso_mask_radius_table(sigma_pix, exptime_s, ccd_x, ccd_y, cam, ccd, sector, prf_path,
                          mags=np.arange(8., 30.01, 0.05)):
    """Mask radius against expected magnitude for one cut, computed once from the PRF at the cut's
    CCD position: r = max(R_MIN, r_iso), r_iso = distance to the outer edge of the furthest pixel
    whose PRF-predicted flux is >= ISO_FRAC sigma_pix, capped at R_MAX."""
    from .prf_grid import get_grid, PRF_SIZE
    prf = get_grid(cam, ccd, sector, prf_path).at(ccd_x, ccd_y)[5, 5]     # centred source, sums to 1
    prf = prf / prf.sum()
    c = PRF_SIZE // 2
    yy, xx = np.mgrid[0:PRF_SIZE, 0:PRF_SIZE]
    dist = np.hypot(xx - c, yy - c).ravel() + 0.5
    order = np.argsort(prf.ravel())[::-1]
    p, d = prf.ravel()[order], np.maximum.accumulate(dist[order])        # furthest pixel among the brightest k
    zp = ZP_600S + 2.5 * np.log10(exptime_s / 600.)
    F = 10 ** (-0.4 * (mags - zp))
    k = np.searchsorted(-p, -ISO_FRAC * sigma_pix / F, side='right')        # pixels with F p >= ISO_FRAC sigma_pix
    r_iso = np.where(k > 0, d[np.maximum(k - 1, 0)], 0.)
    return mags, np.clip(np.maximum(R_MIN, r_iso), R_MIN, R_MAX)


def build_sso_mask(ephemeris_df, reduced_mjd, shape, sigma_pix, exptime_s, ccd_x0, ccd_y0, cam, ccd,
                   sector, prf_path):
    """bool (T, Y, X) cube, True where a predicted asteroid's light falls in that reduced frame.

    ephemeris_df : a cut's predicted ephemeris (x, y cut-local; mjd; mag_expected; designation),
        matched to the reduced frame list here -- its 'frame' column indexes the raw cut TPF,
        and the reduction drops quality-flagged frames.
    reduced_mjd : the reduction's frame times (tessreduce.mjd).
    shape : (T, Y, X) of the reduced cube.
    sigma_pix : difference-image pixel noise (measure_sigma_pix).
    exptime_s : exposure per frame (s), for the zeropoint and the motion within the exposure.
    ccd_x0, ccd_y0 : the cut's corner on the CCD (as asteroid_lightcurves passes to the PSF fit)."""
    T, ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    if len(ephemeris_df) == 0:
        return mask
    eph = ephemeris_df.sort_values(['designation', 'mjd']).reset_index(drop=True)

    # motion over one exposure from the neighbouring ephemeris rows of the same object
    g = eph.groupby('designation', sort=False)
    t = eph['mjd'].to_numpy(float)
    vx, vy = np.zeros(len(eph)), np.zeros(len(eph))
    for _, ix in g.indices.items():
        if len(ix) > 1:
            vx[ix] = np.gradient(eph['x'].to_numpy(float)[ix], t[ix])
            vy[ix] = np.gradient(eph['y'].to_numpy(float)[ix], t[ix])
    half_exp = 0.5 * exptime_s / 86400.
    eph['hx'], eph['hy'] = vx * half_exp, vy * half_exp

    eph = match_ephemeris_to_reduced_frames(eph, reduced_mjd)
    mags, radii = sso_mask_radius_table(sigma_pix, exptime_s, ccd_x0 + nx / 2, ccd_y0 + ny / 2,
                                        cam, ccd, sector, prf_path)
    r = np.interp(eph['mag_expected'].to_numpy(float), mags, radii)
    x, y = eph['x'].to_numpy(float), eph['y'].to_numpy(float)
    hx, hy = eph['hx'].to_numpy(float), eph['hy'].to_numpy(float)
    f = eph['frame'].to_numpy(int)
    ext = r + np.hypot(hx, hy)
    keep = np.isfinite(x) & np.isfinite(y) & (x + ext >= 0) & (x - ext <= nx - 1) & (y + ext >= 0) & (y - ext <= ny - 1)
    for i in np.flatnonzero(keep):
        x0, x1 = max(int(np.floor(x[i] - ext[i])), 0), min(int(np.ceil(x[i] + ext[i])) + 1, nx)
        y0, y1 = max(int(np.floor(y[i] - ext[i])), 0), min(int(np.ceil(y[i] + ext[i])) + 1, ny)
        py, px = np.mgrid[y0:y1, x0:x1]
        # distance from each pixel centre to the segment (x - hx, y - hy) -> (x + hx, y + hy)
        dx, dy = px - (x[i] - hx[i]), py - (y[i] - hy[i])
        L2 = 4 * (hx[i] ** 2 + hy[i] ** 2)
        s = np.clip((dx * 2 * hx[i] + dy * 2 * hy[i]) / L2, 0, 1) if L2 > 0 else 0.
        mask[f[i], y0:y1, x0:x1] |= np.hypot(dx - s * 2 * hx[i], dy - s * 2 * hy[i]) < r[i]
    return mask


def reduce_with_sso_mask(tessreduce, asteroids_path, cam, ccd, sector, cut_corner, prf_path, verbose=1):
    """Run tessreduce.reduce() (constructed with reduce=False) with the cut's predicted asteroids
    excluded from the background. Without predictions it warns and reduces as before."""
    if not table_exists(asteroids_path):
        print(f'!!!WARNING!!! no asteroid predictions at {asteroids_path} -- reducing WITHOUT an SSO '
              'mask, so the background will absorb asteroid flux. Run predict_asteroids first.')
        tessreduce.reduce()
        return None
    ephemeris = load_table(asteroids_path)
    flux = tessreduce.flux_raw if tessreduce.flux_raw is not None else tessreduce.flux
    flux = np.asarray(getattr(flux, 'value', flux))
    sigma_pix = measure_sigma_pix(flux)
    exptime_s = float(np.nanmedian(np.diff(tessreduce.mjd))) * 86400.
    mask = build_sso_mask(ephemeris, tessreduce.mjd, flux.shape, sigma_pix, exptime_s,
                          cut_corner[0], cut_corner[1], cam, ccd, sector, prf_path)
    if verbose > 0:
        print(f'SSO mask: {ephemeris["designation"].nunique() if len(ephemeris) else 0} tracks, '
              f'sigma_pix {sigma_pix:.2f}, exposure {exptime_s:.0f} s, '
              f'{100 * mask.mean():.2f} % of pixel-frames masked ({mask.nbytes / 1e6:.0f} MB)')
    tessreduce.reduce(moving_mask=mask)
    return mask
