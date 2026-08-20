"""
Forced photometry, detrending, star-contamination flagging, and
shift-and-stack for the asteroid tracks predicted by asteroid_prediction.py.

Runs as a pipeline stage after reduce() (needs the reduced flux cube),
consuming a cut's own predicted ephemeris (positions already in cut-local
pixel coordinates) plus its already-cached local Gaia catalogue -- no new
network queries needed for either input.

Two independent photometry methods are computed per predicted position:
  - forced aperture photometry (simple, robust to PRF-model mismatch)
  - forced PSF photometry (matches the flux convention of tessreduce's own
    zeropoint calibration -- see psf_flux_calibration.py's _fit_star, a
    PRF scene-fit, not an aperture sum)
Both matter: aperture photometry is more forgiving of a poorly-matched PRF
near a bright star's halo, while PSF photometry is much better at rejecting
that same halo contamination once pixel-phase systematics are removed (see
detrend_pixel_phase) -- comparing the two is itself a useful diagnostic.
"""
import math
import numpy as np
import pandas as pd

APERTURE_RADIUS_PX = 1.5
STAMP_SIZE = 9
GAIA_MAG_LIMIT = 19.0


# ---------------------------------------------------------------------------
# Forced aperture photometry
# ---------------------------------------------------------------------------

def forced_aperture_photometry(cube, ephemeris_df, radius_px=APERTURE_RADIUS_PX):
    """Forced circular-aperture photometry at every predicted (frame, x, y)
    in ephemeris_df, with local sky background from a concentric annulus
    (sigma-clipped to reject nearby real sources). x, y must already be in
    the cube's own local pixel frame and ephemeris_df must carry a 'frame'
    column indexing directly into cube's first axis (see
    match_ephemeris_to_reduced_frames for building this against the
    REDUCED cube's own frame list, which need not match the raw cut's)."""
    from astropy.stats import sigma_clip
    from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

    r_in, r_out = 2 * radius_px + 3, 2 * radius_px + 7
    rows = []
    for row in ephemeris_df.itertuples():
        if not (0 <= row.x < cube.shape[2] and 0 <= row.y < cube.shape[1]):
            continue
        frame = cube[row.frame]
        aperture = CircularAperture([(row.x, row.y)], radius_px)
        annulus = CircularAnnulus([(row.x, row.y)], r_in, r_out)
        mask = sigma_clip(frame, masked=True, sigma=5).mask
        sky = ApertureStats(frame, annulus, mask=mask)
        phot = aperture_photometry(frame, aperture)
        npix = aperture.area
        sky_mean, sky_std = float(sky.mean[0]), float(sky.std[0])
        flux = float(phot["aperture_sum"].value[0]) - npix * sky_mean
        sig = flux / (np.sqrt(npix) * sky_std) if sky_std and np.isfinite(sky_std) and sky_std > 0 else np.nan
        rows.append(dict(designation=row.designation, frame=int(row.frame), mjd=row.mjd,
                          x=row.x, y=row.y, flux=flux, sky_std=sky_std, sig=sig))
    # pd.DataFrame([]) has no columns at all (not even 'x') -- a fully out-of-bounds track
    # (e.g. every frame lands within STAMP_SIZE/2 of a cut's edge) would otherwise silently
    # break every downstream column access instead of just contributing zero rows
    if not rows:
        return pd.DataFrame(columns=["designation", "frame", "mjd", "x", "y", "flux", "sky_std", "sig"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forced PSF photometry
# ---------------------------------------------------------------------------

def forced_psf_photometry(cube, ephemeris_df, sector, cam, ccd, ccd_x0, ccd_y0,
                            prf_path=None, stamp_size=STAMP_SIZE):
    """Forced PSF photometry (joint fit of [target PRF shape, flat local
    background]) at every predicted (frame, x, y), reusing the same
    per-frame linear solve psf_flux_calibration.py's zeropoint calibration
    uses (_psf_lc_core), so the resulting flux is directly comparable to
    the cut's own AB zeropoint without an aperture correction.

    ccd_x0, ccd_y0 : the cut's own corner offset (full-CCD pixels), needed
    only to look up the right PRF model for this part of the CCD -- x, y
    in ephemeris_df stay in the cube's local frame throughout."""
    from .psf_flux_calibration import _psf_lc_core, PRF_PATH_DEFAULT

    prf_path = prf_path or PRF_PATH_DEFAULT
    half = stamp_size // 2
    rows = []
    for row in ephemeris_df.itertuples():
        xi, yi = int(round(row.x)), int(round(row.y))
        if xi - half < 0 or yi - half < 0 or xi + half + 1 > cube.shape[2] or yi + half + 1 > cube.shape[1]:
            continue
        stamp = cube[row.frame:row.frame + 1, yi - half:yi + half + 1, xi - half:xi + half + 1]
        ccd_x, ccd_y = ccd_x0 + row.x, ccd_y0 + row.y
        try:
            out = _psf_lc_core(stamp, [row.mjd], row.x - xi, row.y - yi, ccd_x, ccd_y,
                                cam, ccd, sector, prf_path=prf_path, stamp_size=stamp_size)
        except Exception:
            continue
        flux, e_flux, bg = float(out["flux_counts"][0]), float(out["e_flux_counts"][0]), float(out["background"][0])
        sig = flux / e_flux if e_flux and np.isfinite(e_flux) and e_flux > 0 else np.nan
        rows.append(dict(designation=row.designation, frame=int(row.frame), mjd=row.mjd,
                          x=row.x, y=row.y, flux=flux, e_flux=e_flux, background=bg, sig=sig))
    # pd.DataFrame([]) has no columns at all (not even 'x') -- a fully out-of-bounds track
    # (e.g. every frame lands within STAMP_SIZE/2 of a cut's edge) or one where every frame's
    # PRF fit raised would otherwise silently break every downstream column access
    if not rows:
        return pd.DataFrame(columns=["designation", "frame", "mjd", "x", "y", "flux", "e_flux", "background", "sig"])
    return pd.DataFrame(rows)


def match_ephemeris_to_reduced_frames(ephemeris_df, reduced_mjd, tol_days=1e-4):
    """Re-derive each ephemeris row's 'frame' index against the REDUCED
    cube's own Times array, by nearest mjd -- the raw cut TPF's frame list
    (what predict_asteroids used) and the reduced product's frame list
    (tessreduce can drop bad-quality frames during reduction) aren't
    guaranteed to match 1:1, so index correspondence can't be assumed.
    Rows with no reduced frame within tol_days are dropped (that exposure
    didn't survive reduction)."""
    reduced_mjd = np.asarray(reduced_mjd)
    order = np.argsort(reduced_mjd)
    sorted_mjd = reduced_mjd[order]
    idx = np.searchsorted(sorted_mjd, ephemeris_df["mjd"].values)
    idx = np.clip(idx, 1, len(sorted_mjd) - 1)
    left, right = sorted_mjd[idx - 1], sorted_mjd[idx]
    use_left = np.abs(ephemeris_df["mjd"].values - left) <= np.abs(right - ephemeris_df["mjd"].values)
    nearest_sorted_idx = np.where(use_left, idx - 1, idx)
    nearest_mjd = sorted_mjd[nearest_sorted_idx]
    within_tol = np.abs(nearest_mjd - ephemeris_df["mjd"].values) <= tol_days

    out = ephemeris_df[within_tol].copy()
    out["frame"] = order[nearest_sorted_idx[within_tol]]
    return out


# ---------------------------------------------------------------------------
# Pixel-phase detrending
# ---------------------------------------------------------------------------

MIN_POINTS_FOR_DETREND = 20
DETREND_SIGMA_CLIP = 3.0
DETREND_N_ITER = 4


def _detrend_design(xfrac, yfrac):
    return np.column_stack([np.ones_like(xfrac), xfrac, yfrac, xfrac * yfrac, xfrac ** 2, yfrac ** 2])


def _detrend_robust_fit(flux, xfrac, yfrac):
    """Fit flux against a full 2nd-order polynomial in sub-pixel phase
    (xfrac, yfrac) -- a bilinear-only model can't represent a radially
    symmetric intra-pixel sensitivity dip, which is what's actually
    observed (flux depends on distance from the nearest pixel centre, not
    x/y direction). Sigma-clipped so a track's own real signal isn't
    absorbed into the phase model."""
    mask = np.isfinite(flux)
    coeffs = None
    for _ in range(DETREND_N_ITER):
        A = _detrend_design(xfrac[mask], yfrac[mask])
        coeffs, *_ = np.linalg.lstsq(A, flux[mask], rcond=None)
        resid = flux[mask] - A @ coeffs
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma <= 0 or not np.isfinite(sigma):
            break
        keep = np.abs(resid - np.median(resid)) < DETREND_SIGMA_CLIP * sigma
        new_mask = mask.copy()
        new_mask[mask] = keep
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    return _detrend_design(xfrac, yfrac) @ coeffs


def detrend_pixel_phase(df, flux_col="flux", e_flux_col="e_flux"):
    """Per-track pixel-phase detrending: removes the modulation (not the
    mean level) of a 2nd-order polynomial fit of flux vs sub-pixel phase.
    Confirmed empirically on real TESS forced-PSF photometry that this is
    a real, non-negligible effect (~30% scatter reduction on a bright,
    otherwise-clean track) -- intra-pixel sensitivity variation that a
    correctly-positioned PRF model alone does not capture. Adds
    flux_detrended / sig_detrended columns; tracks with fewer than
    MIN_POINTS_FOR_DETREND rows are passed through unchanged."""
    out = df.copy()
    out["xfrac"] = out["x"] - np.round(out["x"])
    out["yfrac"] = out["y"] - np.round(out["y"])
    out["flux_detrended"] = out[flux_col]

    for designation, idx in out.groupby("designation").groups.items():
        idx = np.asarray(idx)
        if len(idx) < MIN_POINTS_FOR_DETREND:
            continue
        flux = out.loc[idx, flux_col].values
        model = _detrend_robust_fit(flux, out.loc[idx, "xfrac"].values, out.loc[idx, "yfrac"].values)
        correction = model - np.nanmean(model)
        out.loc[idx, "flux_detrended"] = flux - correction

    if e_flux_col in out.columns:
        out["sig_detrended"] = np.where(out[e_flux_col] > 0, out["flux_detrended"] / out[e_flux_col], np.nan)
    return out


# ---------------------------------------------------------------------------
# Star-contamination flagging
# ---------------------------------------------------------------------------

GMAG_REF = 10.0
RADIUS_REF_PX = 4.0
RADIUS_MIN_PX = 2.0
RADIUS_MAX_PX = 20.0
LOCAL_EXCESS_WINDOW = 151
LOCAL_EXCESS_NSIGMA = 3.0


def flag_radius_px(mag):
    """Brightness-scaled proximity radius (~sqrt(flux) law, anchored at a
    genuinely bright-star threshold) -- validated directionally against
    several real contamination cases at increasing distance/faintness."""
    r = RADIUS_REF_PX * 10 ** (-0.2 * (mag - GMAG_REF))
    return np.clip(r, RADIUS_MIN_PX, RADIUS_MAX_PX)


def local_gaia_cat_to_stars(gaia_cat, wcs, ccd_x0, ccd_y0):
    """Converts the cut's already-cached local_gaia_cat.csv (ra, dec, Gmag,
    Source -- see catalog_queries.create_external_gaia_cat) into cut-local
    pixel positions, avoiding a fresh Gaia query for star flagging. Drops
    any star with a NaN Gmag (real Gaia data: ~0.4% of a typical cut's
    catalog) -- flag_star_contamination can't score an unratable star's
    exclusion radius anyway, and leaving it in poisons its vectorized
    argmin (a NaN flag_radius_px propagates into every distance-margin
    comparison, so argmin can get stuck returning that star's index
    instead of the true closest/worst one -- confirmed: this silently
    hid a real, obvious 0.55px/Gmag=9.76 contaminating star)."""
    gaia_cat = gaia_cat.dropna(subset=["Gmag"])
    x, y = wcs.all_world2pix(gaia_cat["ra"].values, gaia_cat["dec"].values, 0)
    return pd.DataFrame({"x": x - ccd_x0, "y": y - ccd_y0, "mag": gaia_cat["Gmag"].values})


def _local_excess_mask(mjd, flux):
    """A point only counts as contaminated if it's a genuine local outlier
    -- proximity to a star alone triggers on plain coincidence in a dense
    field even for a clean, bright, real track (confirmed: ~23% of one
    real object's frames pass near *some* star purely by chance with zero
    photometric effect, since its own signal is periodic and revisits many
    phases/positions). The window must be well wider than any real
    contamination event's duration (~15-64 frames observed) or the rolling
    median gets pulled up by the event itself, hiding it as its own
    baseline."""
    order = np.argsort(mjd)
    s_sorted = pd.Series(flux[order])
    baseline = s_sorted.rolling(LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median()
    resid = s_sorted - baseline
    mad = (resid - resid.rolling(LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median()).abs().rolling(
        LOCAL_EXCESS_WINDOW, center=True, min_periods=5).median() * 1.4826
    excess = resid.abs() > LOCAL_EXCESS_NSIGMA * mad
    excess_orig_order = np.empty(len(flux), dtype=bool)
    excess_orig_order[order] = excess.values
    return excess_orig_order


def _expand_flag_over_excess_runs(mjd, local_excess, proximity):
    """A real contamination event is one continuous physical crossing --
    the object's distance to the causing star varies smoothly over the
    event, so the strict per-point proximity radius can legitimately be
    satisfied for only part of a contiguous excess run. Once any point in
    a run has confirmed proximity, the whole run is flagged rather than
    only the sub-part that individually clears the radius."""
    order = np.argsort(mjd)
    excess_sorted = local_excess[order]
    prox_sorted = proximity[order]
    flagged_sorted = np.zeros(len(mjd), dtype=bool)

    run_start = None
    for i in range(len(excess_sorted) + 1):
        in_run = i < len(excess_sorted) and excess_sorted[i]
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            if prox_sorted[run_start:i].any():
                flagged_sorted[run_start:i] = True
            run_start = None

    flagged = np.empty(len(mjd), dtype=bool)
    flagged[order] = flagged_sorted
    return flagged


def flag_star_contamination(df, stars, flux_col="flux"):
    """near_bright_star is True only when BOTH hold: a cataloged star falls
    within its own brightness-scaled radius (proximity) AND the light
    curve shows a genuine local excess there (see _local_excess_mask) --
    proximity alone is not sufficient (see _local_excess_mask's
    docstring). Adds contaminating_star_dist_px / _mag and
    near_bright_star columns."""
    out = df.copy()
    if len(stars) == 0:
        # no known stars for this cut (e.g. no local_gaia_cat.csv) -- nothing can be flagged as
        # near one, but np.argmin(..., axis=1) below would otherwise raise on the empty axis
        out["near_star_proximity"] = False
        out["contaminating_star_dist_px"] = np.nan
        out["contaminating_star_mag"] = np.nan
    else:
        sx, sy, smag = stars["x"].values, stars["y"].values, stars["mag"].values
        sr = flag_radius_px(smag)
        d = np.hypot(df["x"].values[:, None] - sx[None, :], df["y"].values[:, None] - sy[None, :])

        margin = d - sr[None, :]
        worst = np.argmin(margin, axis=1)
        out["near_star_proximity"] = margin[np.arange(len(df)), worst] < 0
        out["contaminating_star_dist_px"] = d[np.arange(len(df)), worst]
        out["contaminating_star_mag"] = smag[worst]

    out["local_flux_excess"] = False
    out["near_bright_star"] = False
    for designation, idx in out.groupby("designation").groups.items():
        idx = np.asarray(idx)
        mjd_g = out.loc[idx, "mjd"].values
        excess_g = _local_excess_mask(mjd_g, out.loc[idx, flux_col].values)
        prox_g = out.loc[idx, "near_star_proximity"].values
        out.loc[idx, "local_flux_excess"] = excess_g
        out.loc[idx, "near_bright_star"] = _expand_flag_over_excess_runs(mjd_g, excess_g, prox_g)

    return out


# ---------------------------------------------------------------------------
# Shift-and-stack
# ---------------------------------------------------------------------------

STACK_SIG_TARGET = 5.0


def _required_n(avg_sig, sig_target):
    if avg_sig is None or not np.isfinite(avg_sig) or avg_sig <= 0:
        return None
    return max(int(np.ceil((sig_target / avg_sig) ** 2)), 1)


def _stack_track(track, n_stack):
    track = track.sort_values("mjd").reset_index(drop=True)
    rows = []
    for bin_index, start in enumerate(range(0, len(track), n_stack)):
        chunk = track.iloc[start:start + n_stack]
        w = 1.0 / chunk["e_flux"].values ** 2
        flux = np.sum(chunk["flux_detrended"].values * w) / np.sum(w)
        e_flux = 1.0 / np.sqrt(np.sum(w))
        mjd_lo, mjd_hi = chunk["mjd"].min(), chunk["mjd"].max()
        rows.append(dict(mjd=chunk["mjd"].mean(), mjd_err=(mjd_hi - mjd_lo) / 2,
                          flux=flux, e_flux=e_flux, sig=flux / e_flux, n_frames=len(chunk),
                          bin_index=bin_index))
    return pd.DataFrame(rows)


def stack_lightcurves(psf_df, sig_target=STACK_SIG_TARGET):
    """Shift-and-stack for tracks not already detected on average. The
    'shift' is done by construction -- every frame's forced PSF flux is
    already measured at that frame's own predicted (moving) position, so
    per-frame measurements are already registered to the object's rest
    frame. What's left is combining enough consecutive (inverse-variance
    weighted) frames to beat the noise down by ~sqrt(N) until the combined
    significance clears sig_target. Frames flagged near_bright_star are
    excluded first -- contaminated frames shouldn't be co-added into a
    detection."""
    clean = psf_df[~psf_df["near_bright_star"]]
    summary_rows = []
    stacked_all = []
    for designation, track in clean.groupby("designation"):
        avg_sig = track["sig_detrended"].mean()
        if avg_sig > sig_target:
            summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                      stacking_needed=False, n_stack=1,
                                      n_frames=len(track), achieved_sig=avg_sig))
            continue

        n_needed = _required_n(avg_sig, sig_target)
        if n_needed is None or n_needed > len(track):
            summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                      stacking_needed=True, n_stack=None,
                                      n_frames=len(track), achieved_sig=np.nan))
            continue

        stacked = _stack_track(track, n_needed)
        stacked["designation"] = designation
        stacked_all.append(stacked)
        summary_rows.append(dict(designation=designation, avg_sig=avg_sig,
                                  stacking_needed=True, n_stack=n_needed,
                                  n_frames=len(track), achieved_sig=stacked["sig"].max()))

    if summary_rows:
        summary = pd.DataFrame(summary_rows).set_index("designation")
    else:
        summary = pd.DataFrame(columns=["designation", "avg_sig", "stacking_needed", "n_stack",
                                          "n_frames", "achieved_sig"]).set_index("designation")
    stacked_df = pd.concat(stacked_all, ignore_index=True) if stacked_all else pd.DataFrame(
        columns=["mjd", "mjd_err", "flux", "e_flux", "sig", "n_frames", "bin_index", "designation"])
    return summary, stacked_df


# ---------------------------------------------------------------------------
# Predicted-vs-measured offset from shift-and-stack images
# ---------------------------------------------------------------------------

STACK_CENTROID_SIG_THRESHOLD = 8.0
STACK_CENTROID_STAMP_HALF = 6
STACK_CENTROID_WINDOW = 2
STACK_CENTROID_MIN_STAMPS = 5


def measure_stack_centroid_offset(track, cube, half=STACK_CENTROID_STAMP_HALF,
                                    window=STACK_CENTROID_WINDOW, min_stamps=STACK_CENTROID_MIN_STAMPS):
    """Predicted-vs-measured offset for one track, from its own shift-and-
    stack image -- the same measurement validated by hand on (45124) 1999
    XS88 and (77999) 2002 JD48. Every frame's stamp is extracted at the
    predicted position (already recorded per-frame in track's x, y, frame
    columns) and sub-pixel shifted so the predicted position lands exactly
    at the stamp centre in every frame; averaging then gives a much
    higher-SNR image than any single frame, on which a real offset shows
    up as a flux-weighted centroid displaced from that centre. Restricted
    to a small window around the centre when measuring, since the full
    stamp is dominated by unrelated background/noise pixels far from the
    real source (confirmed: an unrelated corner noise spike shifted a
    whole-stamp centroid by several px on a real case).

    Returns (offset_x, offset_y, n_stamps) -- offset is NaN if fewer than
    min_stamps frames had a usable in-bounds stamp."""
    from scipy.ndimage import shift as ndshift

    stamps = []
    for row in track.itertuples():
        xi, yi = int(round(row.x)), int(round(row.y))
        if xi - half < 0 or yi - half < 0 or xi + half + 1 > cube.shape[2] or yi + half + 1 > cube.shape[1]:
            continue
        stamp = cube[row.frame, yi - half:yi + half + 1, xi - half:xi + half + 1]
        dx, dy = row.x - xi, row.y - yi
        stamps.append(ndshift(stamp, (-dy, -dx), order=1, mode="nearest"))

    if len(stamps) < min_stamps:
        return np.nan, np.nan, len(stamps)

    stacked_mean = np.mean(stamps, axis=0)
    sub = stacked_mean[half - window:half + window + 1, half - window:half + window + 1]
    yy, xx = np.mgrid[half - window:half + window + 1, half - window:half + window + 1]
    w = np.clip(sub - np.median(stacked_mean), 0, None)
    if w.sum() <= 0:
        return np.nan, np.nan, len(stamps)
    cx = (xx * w).sum() / w.sum()
    cy = (yy * w).sum() / w.sum()
    return cx - half, cy - half, len(stamps)


def pool_offset_from_stacks(psf_df, stack_summary, cube, sig_threshold=STACK_CENTROID_SIG_THRESHOLD,
                              min_tracks=2, **stamp_kwargs):
    """Robust per-cut (dx, dy) offset, pooled (median) from the individual
    shift-and-stack centroid offsets (measure_stack_centroid_offset) of
    every track that reached at least sig_threshold -- these are the only
    tracks whose position is measured precisely enough to trust
    individually, unlike a single detected-source centroid. Falls back to
    (nan, nan, 0) if fewer than min_tracks qualify, so callers can fall
    back to a different estimate (e.g. asteroid_photometry.identify_known_asteroids's
    own detected-source pooling) rather than trust too few points."""
    clean = psf_df[~psf_df["near_bright_star"]]
    offsets_x, offsets_y = [], []
    for designation, track in clean.groupby("designation"):
        row = stack_summary.loc[designation] if designation in stack_summary.index else None
        achieved_sig = row["achieved_sig"] if row is not None else np.nan
        if not np.isfinite(achieved_sig) or achieved_sig < sig_threshold:
            continue
        ox, oy, n_stamps = measure_stack_centroid_offset(track, cube, **stamp_kwargs)
        if np.isfinite(ox):
            offsets_x.append(ox)
            offsets_y.append(oy)

    if len(offsets_x) < min_tracks:
        return np.nan, np.nan, len(offsets_x)
    return float(np.median(offsets_x)), float(np.median(offsets_y)), len(offsets_x)


# ---------------------------------------------------------------------------
# Expected apparent magnitude (for reporting alongside measured photometry)
# ---------------------------------------------------------------------------

def measured_apparent_magnitude(flux, zp, sig_threshold=3.0, sig=None):
    """Median apparent magnitude from detections above sig_threshold (if
    sig is given), else from all positive flux -- for reporting alongside
    each track's mag_expected (see asteroid_prediction.expected_apparent_magnitude)."""
    flux = np.asarray(flux, dtype=float)
    if sig is not None:
        flux = flux[np.asarray(sig) > sig_threshold]
    flux = flux[flux > 0]
    if len(flux) == 0:
        return np.nan
    return zp - 2.5 * math.log10(np.median(flux))


# ---------------------------------------------------------------------------
# Cross-matching TESSELLATE's own detections against known asteroids
# ---------------------------------------------------------------------------

MATCH_MAX_DIST_PX = 3.0
MATCH_OFFSET_SEARCH_PX = 5.0
MATCH_MIN_OFFSET_PAIRS = 5


def _pool_cut_offset(detected_df, pred_groups, offset_search_px, min_offset_pairs,
                       detected_id_col, detected_x_col, detected_y_col):
    """Robust per-cut (dx, dy) offset estimate, pooled from every
    (detected, predicted) pair within offset_search_px across the WHOLE
    cut -- not fit per detected object. A single asteroid detection only
    has a handful of noisy points, nowhere near enough to trust a fitted
    offset from; pooling across every object in the cut gives far more
    data for the same underlying systematic (ephemeris/WCS residual is a
    property of the cut/epoch, not of any one object). Falls back to
    zero offset if too few pairs fall within the search radius to trust
    a pooled estimate."""
    all_dx, all_dy = [], []
    for _, det in detected_df.groupby(detected_id_col):
        det = det.sort_values("frame")
        for pred in pred_groups.values():
            merged = det.merge(pred, on="frame", how="inner")
            if len(merged) == 0:
                continue
            dx = merged[detected_x_col].values - merged["x"].values
            dy = merged[detected_y_col].values - merged["y"].values
            close = np.hypot(dx, dy) <= offset_search_px
            all_dx.extend(dx[close])
            all_dy.extend(dy[close])

    if len(all_dx) < min_offset_pairs:
        return 0.0, 0.0
    return float(np.median(all_dx)), float(np.median(all_dy))


def identify_known_asteroids(detected_df, predicted_df, max_dist_px=MATCH_MAX_DIST_PX,
                               offset_search_px=MATCH_OFFSET_SEARCH_PX,
                               min_offset_pairs=MATCH_MIN_OFFSET_PAIRS,
                               detected_id_col="objid",
                               detected_x_col="xcentroid", detected_y_col="ycentroid",
                               offset_x=None, offset_y=None):
    """Proximity cross-match between TESSELLATE's own detections and
    predicted asteroid positions at the same frame, correcting for the
    real systematic offset between predicted and measured position
    (confirmed empirically, up to ~1px even for a validated real object)
    -- without fitting that offset per detected object, since a single
    asteroid detection's few, noisy points aren't reliable enough for
    that.

    offset_x/offset_y let the caller pass in a precomputed cut offset --
    intended for pool_offset_from_stacks' shift-and-stack image centroid
    measurement, which is far more precise than anything derivable from
    TESSELLATE's own noisy per-frame detected centroids (the two tracks it
    was validated on, 45124 and 77999, both had SNR too high for a
    detected-centroid fit to compete). If left as None (the default), the
    offset falls back to _pool_cut_offset's raw-detected-position pooling
    across the whole cut instead.

    For each detected object, checks every frame it shares with each
    candidate predicted asteroid (after shifting the prediction by the
    cut offset) and keeps the single closest approach; a detected
    object counts as matched if that closest approach is within
    max_dist_px, and the closest match overall wins if more than one
    asteroid clears the threshold.

    Returns one row per detected object: the best-matching designation
    (None if nothing cleared max_dist_px), the matching distance, the
    frame it occurred at, and the cut_offset_x/y actually applied
    (same for every row -- useful for sanity-checking the correction).
    Join back onto detected_sources/detected_events/detected_objects by
    detected_id_col."""
    pred_groups = {name: g.sort_values("frame")[["frame", "x", "y"]]
                   for name, g in predicted_df.groupby("designation")}

    if offset_x is None or offset_y is None or not (np.isfinite(offset_x) and np.isfinite(offset_y)):
        offset_x, offset_y = _pool_cut_offset(detected_df, pred_groups, offset_search_px, min_offset_pairs,
                                                detected_id_col, detected_x_col, detected_y_col)

    rows = []
    for obj_id, det in detected_df.groupby(detected_id_col):
        det = det.sort_values("frame")
        best = None
        for designation, pred in pred_groups.items():
            merged = det.merge(pred, on="frame", how="inner")
            if len(merged) == 0:
                continue

            dist = np.hypot(merged[detected_x_col].values - (merged["x"].values + offset_x),
                             merged[detected_y_col].values - (merged["y"].values + offset_y))
            i = int(np.argmin(dist))
            min_dist = float(dist[i])
            if min_dist > max_dist_px:
                continue

            candidate = dict(designation=designation, dist_px=min_dist,
                              frame=int(merged["frame"].values[i]))
            if best is None or candidate["dist_px"] < best["dist_px"]:
                best = candidate

        row = {detected_id_col: obj_id}
        if best is not None:
            row.update(best)
            row["matched_known_asteroid"] = True
        else:
            row.update(dict(designation=None, dist_px=np.nan, frame=None))
            row["matched_known_asteroid"] = False
        row["cut_offset_x"] = offset_x
        row["cut_offset_y"] = offset_y
        rows.append(row)

    return pd.DataFrame(rows)
