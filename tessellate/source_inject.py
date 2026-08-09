from scipy.stats import exponnorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.spatial import cKDTree
from scipy.ndimage import shift
from astropy.stats import sigma_clipped_stats
import os
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns

from tessellate.tessellate import Tessellate
from tessellate.dataprocessor import DataProcessor
from tessellate.navigator import Navigator
from tessellate.tools import RoundToInt, _Print_buff

def _emg_placement(K_internal, duty_frac, threshold=0.05, margin=0.05, n_fine=5000):
    """
    Finds (loc, scale) so that the EMG shape's above-`threshold` region sits
    at tau in [margin, margin + target_width] on the normalized [0,1] domain,
    where target_width = duty_frac * (1 - 2*margin).

    Only used to calibrate the shape's placement once per event - the fine
    grid here is on an abstract, resolution-independent tau axis and has
    nothing to do with real cadence/gaps, so it can't reintroduce the old bug.
    """
    ref_tau = np.linspace(-5, 5 + 10 * K_internal, n_fine)
    ref_flux = exponnorm.pdf(ref_tau, K=K_internal, loc=0, scale=1)
    ref_flux /= ref_flux.max()

    above = ref_flux >= threshold
    if not above.any():
        raise ValueError("threshold too high for given K - no region above it")
    ref_width = ref_tau[above].max() - ref_tau[above].min()
    ref_left = ref_tau[above].min()

    target_width = duty_frac * (1 - 2 * margin)
    scale = target_width / ref_width
    loc = margin - ref_left * scale
    return loc, scale, target_width


def _emg_bin_flux(edges_lo, edges_hi, K_internal, loc, scale, mirror):
    """
    Exact analytic integral of the (possibly mirrored) EMG pdf over each
    [edges_lo, edges_hi] bin, via the exponnorm CDF. Works for arbitrarily
    irregular bin spacing - no interpolation, no fine grid needed here.
    """
    if mirror:
        # mirrored shape g(tau) = f(1-tau); integral over [a,b] of g
        # = integral over [1-b,1-a] of f = F(1-a) - F(1-b)
        cdf_lo = exponnorm.cdf(1 - edges_lo, K=K_internal, loc=loc, scale=scale)
        cdf_hi = exponnorm.cdf(1 - edges_hi, K=K_internal, loc=loc, scale=scale)
        return cdf_lo - cdf_hi
    else:
        cdf_lo = exponnorm.cdf(edges_lo, K=K_internal, loc=loc, scale=scale)
        cdf_hi = exponnorm.cdf(edges_hi, K=K_internal, loc=loc, scale=scale)
        return cdf_hi - cdf_lo


def Flare_Shape(K, duty_frac=0.3, n_fine=2000, threshold=0.05, margin=0.05):
    """
    Kept for previewing/plotting a shape on a regular tau grid - no longer
    used internally by Gen_Event (which evaluates the exact analytic CDF
    directly at real timestamps instead).
    """
    K = max(K, 1e-6)
    loc, scale, _ = _emg_placement(K, duty_frac, threshold, margin, n_fine)
    tau_grid = np.linspace(0, 1, n_fine)
    raw = exponnorm.pdf(tau_grid, K=K, loc=loc, scale=scale)
    flux_norm = raw / raw.max()
    return tau_grid, flux_norm

def Gen_Event(K, cadence_min, time_days, event_time_min, duty_frac=0.6,
              threshold=0.05, margin=0.05, n_fine=5000):
    """
    Evaluates the flare/dip shape exactly (analytic EMG CDF) at the actual
    observation times - no fine grid, no interpolation, no assumption of
    regular spacing.

    K: shape in [-1, 1]
    cadence_min: nominal exposure duration of one frame, minutes (bin width)
    time_days: actual MJD timestamps to generate flux for. These should
               already be just the "active" (above-threshold) window - e.g.
               self.nav.time[frame_start:frame_end] as chosen by the
               gap-aware scheduler below.
    event_time_min: the INTENDED active-region duration in minutes (what
               draw_duration_days() drew) - NOT the padded window, and NOT
               shortened by any gap truncation. This must match what was
               used at scheduling time so the shape placement agrees.
    duty_frac, threshold, margin: must match scheduling-time values.

    Returns:
        flux: array, same length as time_days, peak abs deviation = 1
    """
    K = np.clip(K, -1, 1)
    K_max = 8.0
    K_internal = abs(K) * K_max
    mirror = K < 0

    loc, scale, target_width = _emg_placement(K_internal, duty_frac, threshold, margin, n_fine)
    total_window_min = event_time_min / duty_frac
    tau_lo = margin if not mirror else 1 - (margin + target_width)

    t_min = (time_days - time_days[0]) * 1440.0
    tau = tau_lo + t_min / total_window_min
    half_w_tau = (cadence_min / 2.0) / total_window_min

    flux = _emg_bin_flux(tau - half_w_tau, tau + half_w_tau, K_internal, loc, scale, mirror)

    peak = np.max(np.abs(flux))
    if peak > 0:
        flux = flux / peak
    return flux


def Gen_Sinusoid(time_days, period_min, cadence_min, phase=0.0):
    """
    Evaluates the sinusoid's cadence-integrated flux exactly (closed-form
    integral of sin) at the actual observation times. Handles any gaps for
    free - missing frames are simply absent from time_days, not extrapolated
    across.

    time_days: actual MJD timestamps to generate flux for (can span the
               full baseline, gaps and all).
    period_min: oscillation period, minutes.
    cadence_min: nominal exposure duration of one frame, minutes (bin width -
                 NOT inferred from spacing between frames; a gap means
                 missing frames, not a wider exposure).
    phase: starting phase (radians), referenced to time_days[0].

    Returns:
        flux: array, same length as time_days, peak abs deviation = 1
    """
    t_min = (time_days - time_days[0]) * 1440.0
    omega = 2 * np.pi / period_min
    half_w = cadence_min / 2.0

    antideriv = lambda t: -np.cos(omega * t + phase) / omega
    flux = antideriv(t_min + half_w) - antideriv(t_min - half_w)

    peak = np.max(np.abs(flux))
    if peak > 0:
        flux = flux / peak
    return flux

def _Shift_One(frame, s):
	if np.nansum(abs(frame)) > 0:
		return shift(frame, [s[0], s[1]], mode='nearest', order=5)
	return frame

class SourceInjector():

    def __init__(self,sector,cam,ccd,n=8,job_output_path='.',working_path='.',num_cores=None,
                 data_path='/fred/oz335/TESSdata',prf_path='/fred/oz335/_local_TESS_PRFs',injection_dir='source_injection'):

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.n = n

        self.cut = None

        self.job_output_path = job_output_path
        self.working_path = working_path
        self.data_path = data_path
        self.prf_path = prf_path
        self.injection_dir = injection_dir

        self.num_cores = multiprocessing.cpu_count() if num_cores is None else num_cores

        self.path = f'{self.data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}'
        self.nav = Navigator(sector,cam,ccd,data_path,n,injection=True,injection_dir=injection_dir)



    def _find_injection_sites(self, min_sep=5, edge_buffer=5, grid_step=1):
        """
        Finds valid pixel locations for source injection, defined as points at
        least `min_sep` pixels from every existing detected object and at least
        `edge_buffer` pixels from the cube edge.

        objects_df: DataFrame with existing detected object positions - expects
                    'x' and 'y' columns (pixel coordinates); adjust names below
                    if your Navigator uses different column labels
        cube_shape: (n_frames, ny, nx) or (ny, nx) of the data cube
        min_sep: minimum allowed distance (pixels) from any existing object
        edge_buffer: minimum allowed distance (pixels) from the cube edge
        grid_step: spacing (pixels) of the candidate grid searched over - 1 for
                full resolution, >1 to speed up the search on large cubes

        Returns:
            valid_sites: array of shape (n_valid, 2), (x, y) integer pixel
                        positions satisfying both constraints
        """

        print('    finding injection sites')

        ny, nx = self.nav.flux.shape[-2], self.nav.flux.shape[-1]

        xs = np.arange(edge_buffer, nx - edge_buffer, grid_step)
        ys = np.arange(edge_buffer, ny - edge_buffer, grid_step)
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)

        obj_xy = self.nav.objects[['xcentroid', 'ycentroid']].dropna().values
        if len(obj_xy) == 0:
            return candidates.astype(int)

        tree = cKDTree(obj_xy)
        dist, _ = tree.query(candidates, k=1)

        valid_sites = candidates[dist >= min_sep].astype(int)
        return valid_sites

    def _frame_window_with_gap_cutoff(self, frame_start, max_duration_days, cadence_min, gap_factor=10.0):
        """
        Walks forward from frame_start through the real self.nav.time array
        and returns frame_end (exclusive) such that:
          - elapsed real time from frame_start doesn't exceed max_duration_days
          - no gap between consecutive included frames exceeds
            gap_factor * cadence_min
        i.e. the window is cut short - not padded or rejected - the moment
        it runs into a big gap or the end of the baseline.
        """
        time = self.nav.time
        n_frames = len(time)
        if frame_start >= n_frames - 1:
            return min(frame_start + 1, n_frames)

        gap_thresh_days = gap_factor * (cadence_min / 1440.0)
        t0 = time[frame_start]

        frame_end = frame_start + 1
        while frame_end < n_frames:
            if time[frame_end] - time[frame_end - 1] > gap_thresh_days:
                break
            if time[frame_end] - t0 > max_duration_days:
                break
            frame_end += 1
        return frame_end


    def schedule_injections(self, valid_sites, n_events,
                        duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
                        K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
                        duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
                        stamp_area=25, max_frame_fill_frac=0.25,
                        overlap_dist_px=5, max_attempts_per_event=200, rng=None,
                        type_probs=(0.7, 0.2, 0.10),
                        K_negative_range=(-0.2, 0.2),
                        period_range_min=(20, None), period_mode_min=120.0,
                        period_concentration=6.0,
                        gap_factor=10.0):
        """
        Randomly schedules n_events injections across space and time, allowing
        overlap (spatial and/or temporal) but capping how much of any single
        frame's pixel area is occupied by injected sources at once. Flags events
        within `overlap_dist_px` of another event during an overlapping time
        window. Assigns a uniform random sub-pixel offset within the chosen pixel.

        Each event is one of three types, drawn according to `type_probs`
        (probabilities for 'flare', 'negative', 'sinusoid' respectively - must
        sum to 1):
            'flare'    : the existing EMG flare/variable-peak shape (positive-going)
            'negative' : a dip - same EMG machinery, but K is drawn from the
                        narrow `K_negative_range` (near-Gaussian) and the
                        injected flux is negative-going
            'sinusoid' : a continuous oscillation, always spanning the FULL
                        TESS temporal baseline (mjd_start/mjd_end = the first/
                        last available timestamps) with hard edges, at a
                        period drawn skewed toward a couple hours but ranging
                        from `period_range_min[0]` up to (potentially) well
                        beyond the baseline length itself

        Duration, K, and duty_frac are each drawn via a Beta(a, b) distribution
        over their respective ranges (Beta(1,1) = uniform), letting you bias
        toward particular regimes by default. Period is drawn via a Beta(a, b)
        parameterized directly by a target mode (`period_mode_min`) and a
        concentration (`period_concentration`), rather than raw (a, b), since
        the period range spans orders of magnitude (20 min to the full
        multi-week baseline) and a fixed (a, b) tuned for a narrow range
        wouldn't reliably land the peak near a couple hours once the range
        changes with the sector's baseline length.

        valid_sites: array (n_sites, 2), (x, y) candidate positions
        n_events: number of events to schedule
        duration_range_min: (min, max) event duration IN MINUTES - used for
                    'flare'/'negative' events only; sinusoids ignore this and
                    always span the full baseline
        duration_skew: (a, b) Beta params on normalized log-duration axis;
                    a < b biases toward shorter events
        K_range: (min, max) shape parameter range, used for 'flare' events
        K_skew: (a, b) Beta params on the positive-K portion, normalized to
                [0,1] then mapped to [0, K_range[1]]; a > b biases toward high K
        p_negative_K: probability a 'flare' event is drawn from the negative K
                    range (i.e. left-skewed shape - NOT the 'negative' type)
        duty_frac_range: (min, max) allowed duty_frac, used for 'flare'/'negative'
        duty_frac_skew: (a, b) Beta params on normalized duty_frac axis;
                        (1,1) = uniform
        stamp_area: approximate PSF footprint (pixels)
        max_frame_fill_frac: max fraction of a frame's pixels occupied at once
        overlap_dist_px: distance (pixels) defining spatial overlap flag
        max_attempts_per_event: retries per event before giving up
        rng: np.random.Generator, or None to create a default one
        type_probs: (p_flare, p_negative, p_sinusoid), must sum to 1
        K_negative_range: (min, max) K range for 'negative' events - kept
                        narrow/near-zero so dips are close to Gaussian
        period_range_min: (min, max) oscillation period IN MINUTES, used for
                    'sinusoid' events. If max is None, it's auto-set to 2x
                    the sector's temporal baseline (in minutes), so some
                    periods extend past the full baseline.
        period_mode_min: the period (minutes) the distribution peaks at -
                    default 120 (2 hr). Must lie within period_range_min.
        period_concentration: >2, controls how peaked the distribution is
                    around period_mode_min. Larger = tighter clustering
                    around the mode; near 2 = closer to uniform-in-log.

        Returns:
            schedule_df: DataFrame with columns:
                eventid, event_type, xcentroid, ycentroid, snr, K, duty_frac,
                period_min, phase, frame_start, frame_end, frame_duration,
                mjd_start, mjd_end, mjd_duration, overlap
        """

        print('    generating injection properties')

        if rng is None:
            rng = np.random.default_rng()

        assert abs(sum(type_probs) - 1.0) < 1e-6, "type_probs must sum to 1"

        n_frames, ny, nx = self.nav.flux.shape
        frame_area = valid_sites.shape[0]
        max_occupied_px = max_frame_fill_frac * frame_area

        min_to_day = 1 / 1440.0
        log_dur_min = np.log10(duration_range_min[0] * min_to_day)
        log_dur_max = np.log10(duration_range_min[1] * min_to_day)

        t_min, t_max = self.nav.time.min(), self.nav.time.max()
        baseline_days = t_max - t_min
        baseline_min = baseline_days * 1440.0
        cadence_min = np.nanmedian(np.diff(self.nav.time)) * 1440

        period_lo, period_hi = period_range_min
        if period_hi is None:
            period_hi = baseline_min * 2.0  # allow periods well past the baseline
        log_per_min = np.log10(period_lo)
        log_per_max = np.log10(period_hi)

        log_mode = np.log10(period_mode_min)
        assert log_per_min <= log_mode <= log_per_max, \
            "period_mode_min must lie within period_range_min"
        u_mode = (log_mode - log_per_min) / (log_per_max - log_per_min)

        k = max(period_concentration, 2.0001)  # keep a,b > 1 so mode formula holds
        a_per = 1 + u_mode * (k - 2)
        b_per = 1 + (1 - u_mode) * (k - 2)

        occupancy = np.zeros(n_frames)

        def draw_event_type():
            return rng.choice(['flare', 'negative', 'sinusoid'], p=type_probs)

        def draw_duration_days():
            u = rng.beta(duration_skew[0], duration_skew[1])
            log_dur = log_dur_min + u * (log_dur_max - log_dur_min)
            return 10 ** log_dur

        def draw_K(event_type):
            if event_type == 'negative':
                return rng.uniform(K_negative_range[0], K_negative_range[1])
            if rng.uniform() < p_negative_K:
                return rng.uniform(K_range[0], 0)
            u = rng.beta(K_skew[0], K_skew[1])
            return u * K_range[1]

        def draw_duty_frac():
            u = rng.beta(duty_frac_skew[0], duty_frac_skew[1])
            return duty_frac_range[0] + u * (duty_frac_range[1] - duty_frac_range[0])

        def draw_period_min():
            u = rng.beta(a_per, b_per)
            log_per = log_per_min + u * (log_per_max - log_per_min)
            return 10 ** log_per

        def draw_snr():
            if rng.random() < 0.60:          # 75% of draws concentrated in 3-10
                return rng.uniform(1,10)
            else:                             # 25% draws from full range (covers tails)
                return 10**(rng.uniform(1, 2))
        
        rows = []
        eventid = 0 
        while eventid < n_events:
            placed = False
            event_type = draw_event_type()
            for _attempt in range(max_attempts_per_event):
                site = valid_sites[rng.integers(0, len(valid_sites))]
                xfrac, yfrac = rng.uniform(0, 1, size=2)
                xcentroid = site[0] + xfrac
                ycentroid = site[1] + yfrac

                K = np.nan
                duty_frac = np.nan
                period_min = np.nan
                phase = np.nan
                event_time_min = np.nan

                if event_type == 'sinusoid':
                    period_min = draw_period_min()
                    phase = rng.uniform(0, 2 * np.pi)
                    frame_start, frame_end = 0, n_frames
                    mjd_start, mjd_end = t_min, t_max
                else:
                    event_time_days = draw_duration_days()
                    K = draw_K(event_type)
                    duty_frac = draw_duty_frac()
                    event_time_min = event_time_days * 1440.0

                    mjd_start_raw = t_min + rng.uniform(0, 1) * (t_max - t_min)
                    frame_start = int(np.searchsorted(self.nav.time, mjd_start_raw, side='left'))
                    if frame_start >= n_frames:
                        continue

                    frame_end = self._frame_window_with_gap_cutoff(
                        frame_start, event_time_days, cadence_min, gap_factor=gap_factor
                    )

                    mjd_start = self.nav.time[frame_start]
                    mjd_end = self.nav.time[frame_end - 1]

                projected = occupancy[frame_start:frame_end] + stamp_area
                if np.any(projected > max_occupied_px):
                    continue

                occupancy[frame_start:frame_end] += stamp_area
                snr = draw_snr()

                conflict = False
                for r in rows:
                    time_overlap = (
                        frame_start < r['frame_end']
                        and frame_end > r['frame_start']
                    )

                    if not time_overlap:
                        continue

                    dist = np.hypot(
                        xcentroid - r['xcentroid'],
                        ycentroid - r['ycentroid']
                    )

                    if dist <= overlap_dist_px:
                        conflict = True
                        break

                if conflict:
                    continue 

                rows.append({
                    'event_type': event_type,
                    'xcentroid': xcentroid,
                    'ycentroid': ycentroid,
                    'snr': snr,
                    'K': K,
                    'duty_frac': duty_frac,
                    'period_min': period_min,
                    'phase': phase,
                    'event_time_min': event_time_min,   # NEW: intended active duration (flare/negative only)
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'frame_duration': frame_end - frame_start,
                    'mjd_start': mjd_start,              # now the ACTUAL timestamp of frame_start
                    'mjd_end': mjd_end,                  # now the ACTUAL timestamp of frame_end-1
                    'mjd_duration': mjd_end - mjd_start,  # now the ACTUAL injected span
                })
                placed = True
                eventid += 1
                break

            if not placed:
                print(f"Warning: event {eventid} ({event_type}) could not be placed within "
                    f"{max_attempts_per_event} attempts under the fill-fraction cap")

        schedule_df = pd.DataFrame(rows)
        return schedule_df

    def inject_sources(self,cut,raw_cube,n_events,
                        min_sep=5,edge_buffer=5,grid_step=1,big_size=15,small_size=5,
                        duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
                        K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
                        duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
                        stamp_area=5, max_frame_fill_frac=0.25,
                        overlap_dist_px=5, max_attempts_per_event=200,
                        type_probs=(0.6, 0.2, 0.2),
                        K_negative_range=(-0.2, 0.2),
                        period_range_min=(20, None), period_mode_min=240.0,
                        period_concentration=3.0):


        from PRF import TESS_PRF


        # -- Generate PRF -- #
        dp = DataProcessor(self.sector,data_path=self.data_path)
        _, cutCentrePx, _, _ = dp.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[cut-1][0]
        row = cutCentrePx[cut-1][1]
        if self.sector < 4:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors1_2_3')
        else:
            prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{self.prf_path}/Sectors4+')
                

        valid_sites = self._find_injection_sites(min_sep,edge_buffer,grid_step)

        injections = self.schedule_injections(valid_sites,n_events,
                                                duration_range_min,duration_skew,
                                                K_range, K_skew, p_negative_K,
                                                duty_frac_range, duty_frac_skew,
                                                stamp_area, max_frame_fill_frac,
                                                overlap_dist_px, max_attempts_per_event,
                                                type_probs=type_probs,
                                                K_negative_range=K_negative_range,
                                                period_range_min=period_range_min,
                                                period_mode_min=period_mode_min,
                                                period_concentration=period_concentration)

        injections['frame_max'] = 0
        injections['mjd_max'] = 0
        cadence_min = np.nanmedian(np.diff(self.nav.time)) * 1440
        lcs = []
        for i in tqdm(range(n_events), desc='    injecting events into cube', position=0, leave=True, dynamic_ncols=False, ascii=True):
            source = injections.iloc[i]

            time_days = self.nav.time[source.frame_start:source.frame_end]
            if len(time_days) == 0:
                continue

            if source.event_type == 'sinusoid':
                flux = Gen_Sinusoid(time_days, source.period_min, cadence_min, phase=source.phase)
                sign = 1.0
            else:
                flux = Gen_Event(source.K, cadence_min, time_days, source.event_time_min, source.duty_frac)
                sign = -1.0 if source.event_type == 'negative' else 1.0

            frames = np.arange(source.frame_start, source.frame_end)
            ref_idx = np.argmax(np.abs(flux))
            max_frame = frames[ref_idx]
            injections.iloc[i, injections.columns.get_loc('frame_max')] = max_frame
            injections.iloc[i, injections.columns.get_loc('mjd_max')] = self.nav.time[max_frame]
            
            xint = RoundToInt(source.xcentroid)
            yint = RoundToInt(source.ycentroid)

            half_big = big_size // 2
            h, w = self.nav.flux.shape[1], self.nav.flux.shape[2]

            y1 = yint - half_big        # Desired bounds in full image
            y2 = yint + half_big + 1
            x1 = xint - half_big
            x2 = xint + half_big + 1
        
            yy1, yy2 = max(0, y1), min(h, y2)   # Clip to image bounds
            xx1, xx2 = max(0, x1), min(w, x2)
        
            cut = np.full((big_size, big_size), np.nan, dtype=np.float32)   # Create NaN-padded cut
    
            cy1 = yy1 - y1
            cy2 = cy1 + (yy2 - yy1)
            cx1 = xx1 - x1
            cx2 = cx1 + (xx2 - xx1)
    
            cut[cy1:cy2, cx1:cx2] = self.nav.flux[max_frame, yy1:yy2, xx1:xx2] 
        
            valid = cut[~np.isnan(cut)]     # Compute noise only on valid pixels
            if valid.size == 0:
                continue
            _, _, noise = sigma_clipped_stats(valid, sigma=3)

            npix = 9
            b = -source.snr**2 / 600
            c = -source.snr**2 * npix * noise**2
            peak_flux = (-b + np.sqrt(b**2 - 4*c)) / 2
            peak_flux *= sign

            flux *= peak_flux
            lcs.append(np.array([frames,flux]))

            image = prf.locate(2 + (source.xcentroid - RoundToInt(source.xcentroid)),
                                2 + (source.ycentroid - RoundToInt(source.ycentroid)),
                                (5, 5))
            
            for j, f in enumerate(flux):

                image_frame = image.copy() * f / np.nansum(image[1:4, 1:4])

                raw_cube[frames[j], yint-2:yint+3, xint-2:xint+3] += image_frame

        injections.reset_index(names='injid',inplace=True)

        return raw_cube,injections,lcs

    def apply_shifts(self,shifts,cube):

        print('    applying shifts')        

        result = Parallel(n_jobs=self.num_cores)(
					delayed(_Shift_One)(cube[i], -1*shifts[i])
					for i in tqdm(range(len(cube)), position=0, leave=True,dynamic_ncols=False,ascii=True))

        return np.array(result)

    def load_raw_cube(self,cut,cube_mode):

        directory = f'{self.path}/Cut{cut}of{self.n**2}'
        base_name = f'sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'  

        if os.path.exists(f'{directory}/{base_name}.fits') and (cube_mode == 'cutfits'):
            print('Loading raw lightkurve TPF')
            import lightkurve as lk
            tpf = lk.TessTargetPixelFile(f'{directory}/{base_name}.fits',quality_bitmask='hard')
            raw_cube = tpf.flux.value
            processed = False

        else:
            raw_cube = self.nav.flux
            processed = True

        return raw_cube,processed

    def run(self,cut,n_events,overwrite=False,cube_mode='cutfits',
            min_sep=5,edge_buffer=5,grid_step=1,big_size=15,small_size=5,
            duration_range_min=(10, 1440), duration_skew=(1.0, 2.5),
            K_range=(-1, 1), K_skew=(1.0, 1.0), p_negative_K=0.05,
            duty_frac_range=(0.05, 0.95), duty_frac_skew=(1.0, 1.0),
            stamp_area=25, max_frame_fill_frac=0.25,
            overlap_dist_px=5, max_attempts_per_event=200,
            type_probs=(0.6, 0.2, 0.2),
            K_negative_range=(-0.2, 0.2),
            period_range_min=(20, None), period_mode_min=240.0,
            period_concentration=3.0):


        _Print_buff(60,f'Running Source Injection for Sector{self.sector} Cam{self.cam} Ccd{self.ccd} Cut {cut}')
        print('\n')

        directory = f'{self.path}/Cut{cut}of{self.n**2}'
        base_name = f'sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'      

        inject = False
        if not os.path.exists(f'{directory}/{self.injection_dir}/{base_name}_RawFlux.npy'): 
            inject = True
        elif overwrite:
            os.system(f'rm -r {directory}/{self.injection_dir}')
            inject = True

        if inject:
            
            self.nav.gather_results(cut=cut,sources=False,events=True,objects=True)
            self.nav.gather_data(cut=cut,flux=True,time=True,bkg=True,verbose=False)
            raw_cube,processed = self.load_raw_cube(cut,cube_mode) 

            raw_cube,injections,lcs = self.inject_sources(cut,raw_cube,n_events,
                        min_sep,edge_buffer,grid_step,big_size,small_size,
                        duration_range_min, duration_skew,
                        K_range, K_skew, p_negative_K,
                        duty_frac_range, duty_frac_skew,
                        stamp_area, max_frame_fill_frac,
                        overlap_dist_px, max_attempts_per_event,
                        type_probs=type_probs,
                        K_negative_range=K_negative_range,
                        period_range_min=period_range_min,
                        period_mode_min=period_mode_min,
                        period_concentration=period_concentration)

            if processed:

                orbit_segments = np.load(f'{directory}/{base_name}_OrbitSegments.npy')
                orbit_refs = np.load(f'{directory}/{base_name}_OrbitRefs.npz')
                orbit_refs = {int(k): orbit_refs[k] for k in orbit_refs.files}
                ref = np.load(f'{directory}/{base_name}_Ref.npy')
                shifts = np.load(f'{directory}/{base_name}_Shifts.npy')

                raw_cube[orbit_segments==1] += orbit_refs[1]
                raw_cube[orbit_segments==2] += orbit_refs[2]
                raw_cube += ref

                raw_cube = self.apply_shifts(shifts,raw_cube)

            lcs_arr = np.empty(len(lcs), dtype=object)
            for i, lc in enumerate(lcs):
                lcs_arr[i] = lc

            os.makedirs(f'{directory}/{self.injection_dir}',exist_ok=True)
            np.savez(f'{directory}/{self.injection_dir}/lightcurves.npz', lcs=lcs_arr)
            injections.to_csv(f'{directory}/{self.injection_dir}/injected_events.csv',index=False)
            np.save(f'{directory}/{self.injection_dir}/{base_name}_RawFlux.npy',raw_cube)

        run = Tessellate(data_path=self.data_path,working_path=self.working_path,job_output_path=self.job_output_path,
                            sector=self.sector,cam=self.cam,ccd=self.ccd,n=self.n,cuts=cut,
                            download=False,make_cube=False,fix_wcs=False,make_cuts=False,calibrate=False,
                            reduce=True,search=True,injection=True,plot=False,delete=False,injection_dir=injection_dir,
                            reset_logs=False,overwrite=False,ask_config=False,save_config=False,use_suggestions=True)



    def match_results_to_transients(self,centroid_match_radius=1.0, min_temporal_iou=0.0, spatial_weight=0.5,
                                    overlap_weight=2.0,duration_weight=0.5,peak_weight=0.1):

        non_vars = self.injections[self.injections.event_type != 'sinusoid']

        # Output columns
        non_vars["detected"] = "n"
        non_vars["ev_match"] = "-"
        non_vars["centroid_sep"] = np.nan
        non_vars["temporal_iou"] = np.nan
        non_vars["duration_ratio"] = np.nan
        non_vars["peak_offset_min"] = np.nan
        non_vars["snr_detected"] = np.nan
        non_vars["snr_ratio"] = np.nan
        non_vars["match_score"] = np.nan

        events_all = self.nav.events.copy()
        frame_bins = np.sort(events_all.frame_bin.dropna().unique())

        for inj_idx, inj in non_vars.iterrows():


            inj_start = float(inj.mjd_start)
            inj_end = float(inj.mjd_end)
            inj_duration = max(inj_end - inj_start, np.finfo(float).eps)

            event_found = False

            # --  Search event catalogues from lowest to highest frame_bin -- #
            for frame_bin in frame_bins:

                bin_events = events_all[
                    events_all.frame_bin == frame_bin
                ].copy()

                # Spatial pre-selection
                bin_events["centroid_sep"] = np.hypot(
                    bin_events.xcentroid - inj.xcentroid,
                    bin_events.ycentroid - inj.ycentroid,
                )

                candidates = bin_events[bin_events.centroid_sep <= centroid_match_radius].copy()
                if len(candidates) == 0:
                    continue

                # Temporal overlap using MJD values
                candidates["intersection"] = np.maximum(
                    0.0,
                    np.minimum(candidates.mjd_end, inj_end)
                    - np.maximum(candidates.mjd_start, inj_start),
                )

                candidates = candidates[candidates.intersection > 0].copy()
                if len(candidates) == 0:
                    continue

                candidates["union"] = (
                    np.maximum(candidates.mjd_end, inj_end)
                    - np.minimum(candidates.mjd_start, inj_start)
                )

                candidates["temporal_iou"] = candidates.intersection / candidates.union

                candidates = candidates[candidates.temporal_iou >= min_temporal_iou].copy()
                if len(candidates) == 0:
                    continue

                candidates["det_duration"] = (
                    candidates.mjd_end - candidates.mjd_start
                ).clip(lower=np.finfo(float).eps)

                candidates["overlap_frac_inj"] = candidates.intersection / inj_duration
                candidates["overlap_frac_det"] = candidates.intersection / candidates.det_duration

                candidates["duration_ratio"] = np.maximum(
                    candidates.det_duration,
                    inj_duration,
                ) / np.minimum(
                    candidates.det_duration,
                    inj_duration,
                )

                candidates["peak_offset_min"] = np.abs(candidates.mjd_max - inj.mjd_max) * 1440.0
            
                # Normalise peak offset by injected duration so long events are
                # not unfairly penalised.
                inj_duration_min = max(inj_duration * 1440.0, 1.0)

                candidates["match_score"] = (
                    spatial_weight * candidates.centroid_sep
                    + overlap_weight * (1.0 - candidates.temporal_iou)
                    + duration_weight * np.abs(
                        np.log(candidates.duration_ratio)
                    )
                    + peak_weight * (
                        candidates.peak_offset_min / inj_duration_min
                    )
                )

                best_idx = candidates.match_score.idxmin()
                best = candidates.loc[best_idx]

                detected_label = "y" if int(frame_bin) == 1 else str(int(frame_bin))

                non_vars.loc[inj_idx, "detected"] = detected_label
                non_vars.loc[inj_idx, "ev_match"] = f"{best.objid}_{best.eventid}"
                non_vars.loc[inj_idx, "centroid_sep"] = best.centroid_sep
                non_vars.loc[inj_idx, "temporal_iou"] = best.temporal_iou
                non_vars.loc[inj_idx, "duration_ratio"] = best.duration_ratio
                non_vars.loc[inj_idx, "peak_offset_min"] = best.peak_offset_min
                non_vars.loc[inj_idx, "snr_detected"] = best.lc_sig_max
                non_vars.loc[inj_idx, "snr_ratio"] = best.lc_sig_max / inj.snr
                non_vars.loc[inj_idx, "match_score"] = best.match_score
                non_vars.loc[inj_idx,"z_xcentroid"] = (best.xcentroid-inj.xcentroid) / best.xcentroid_err
                non_vars.loc[inj_idx,"z_ycentroid"] = (best.ycentroid-inj.ycentroid) / best.ycentroid_err

                event_found = True
                break

            if event_found:
                continue

            # -- No event found: check isolated single-frame detections -- #
            isolated = self.nav.isolated.copy()

            isolated["centroid_sep"] = np.hypot(
                isolated.xcentroid - inj.xcentroid,
                isolated.ycentroid - inj.ycentroid,
            )

            # Injection frame_end is assumed exclusive.
            iso_candidates = isolated[
                (isolated.frame >= int(inj.frame_start))
                & (isolated.frame <= int(inj.frame_end))
                & (isolated.centroid_sep <= centroid_match_radius)
            ].copy()

            if len(iso_candidates) == 0:
                continue

            non_vars.loc[inj_idx, "detected"] = "iso"

        return non_vars


    # def match_vars_to_injections(self,centroid_match_radius=1.0,extremum_window_frac=0.25,min_samples_per_extremum=1):

    #     vars = self.injections[self.injections.event_type == 'sinusoid']

    #     # Output columns
    #     columns = {
    #         "detected" : "n",
    #         "obj_match": np.nan,
    #         "centroid_sep": np.nan,
    #         "n_object_events": 0,
    #         "n_expected_extrema": 0,
    #         "n_detected_extrema": 0,
    #         "var_completeness": np.nan,
    #         "var_purity": np.nan,
    #         "var_phase_rms": np.nan,
    #     }

    #     for column, default in columns.items():
    #         vars[column] = default

    #     nav_time = np.asarray(self.nav.time)

    #     objects = self.nav.objects[self.nav.objects.frame_bin == 1].copy()

    #     for inj_idx,inj in vars.iterrows():

    #         object_sep = np.hypot(objects.xcentroid - inj.xcentroid,
    #                               objects.ycentroid - inj.ycentroid)

    #         nearby = objects[object_sep <= centroid_match_radius].copy()

    #         if len(nearby) == 0:
    #             continue

    #         nearby["centroid_sep"] = object_sep[object_sep <= centroid_match_radius]

    #         best_object = nearby.loc[nearby.centroid_sep.idxmin()]

    #         events = self.nav.events[self.nav.events.objid == best_object.objid].copy()

    #         self.injections.loc[inj_idx, "obj_match"] = best_object.objid
    #         self.injections.loc[inj_idx, "centroid_sep"] = best_object.centroid_sep
    #         self.injections.loc[inj_idx, "n_object_events"] = len(events)

    #         period_days = inj.period_min / 1440.0
    #         half_period_days = period_days / 2.0

    #         start = float(inj.mjd_start)
    #         end = float(inj.mjd_end)

    #         first_extremum_offset = (
    #             (np.pi / 2.0 - inj.phase)
    #             / (2.0 * np.pi)
    #             * period_days
    #         )

    #         # Shift the first extremum forward until it lies within the
    #         # injection interval.
    #         k_start = int(
    #             np.ceil(
    #                 (start - (start + first_extremum_offset))
    #                 / half_period_days
    #             )
    #         )

    #         first_extremum = (
    #             start
    #             + first_extremum_offset
    #             + k_start * half_period_days
    #         )

    #         if first_extremum > end:
    #             self.injections.loc[inj_idx, "detected"] = "n"
    #             continue

    #         n_extrema = (
    #             int(np.floor((end - first_extremum) / half_period_days))
    #             + 1
    #         )

    #         extrema_times = (
    #             first_extremum
    #             + np.arange(n_extrema) * half_period_days
    #         )

    #         # Window half-width around each expected extremum.
    #         extremum_half_width = (
    #             extremum_window_frac * half_period_days
    #         )

    #         extrema_window_start = (
    #             extrema_times - extremum_half_width
    #         )

    #         extrema_window_end = (
    #             extrema_times + extremum_half_width
    #         )

    #         # --------------------------------------------------------------
    #         # Remove extrema that are too poorly sampled
    #         # --------------------------------------------------------------
    #         if min_samples_per_extremum > 1:

    #             sample_counts = np.array([
    #                 np.sum(
    #                     (nav_time >= window_start)
    #                     & (nav_time <= window_end)
    #                 )
    #                 for window_start, window_end in zip(
    #                     extrema_window_start,
    #                     extrema_window_end,
    #                 )
    #             ])

    #             sampled = (
    #                 sample_counts >= min_samples_per_extremum
    #             )

    #             extrema_times = extrema_times[sampled]
    #             extrema_window_start = extrema_window_start[sampled]
    #             extrema_window_end = extrema_window_end[sampled]

    #         n_expected = len(extrema_times)

    #         self.injections.loc[
    #             inj_idx,
    #             "n_expected_extrema",
    #         ] = n_expected

    #         if n_expected == 0:
    #             self.injections.loc[inj_idx, "detected"] = "n"
    #             continue

    #         if len(events) == 0:
    #             self.injections.loc[inj_idx, "detected"] = "n"
    #             self.injections.loc[inj_idx, "n_detected_extrema"] = 0
    #             self.injections.loc[inj_idx, "n_matched_events"] = 0
    #             self.injections.loc[inj_idx, "n_spurious_events"] = 0
    #             self.injections.loc[inj_idx, "var_completeness"] = 0.0
    #             self.injections.loc[inj_idx, "var_purity"] = np.nan
    #             continue

    #         # --------------------------------------------------------------
    #         # Determine which events overlap which extremum windows
    #         # --------------------------------------------------------------
    #         event_start = events.mjd_start.to_numpy()
    #         event_end = events.mjd_end.to_numpy()

    #         overlap_matrix = (
    #             event_start[:, None] < extrema_window_end[None, :]
    #         ) & (
    #             event_end[:, None] > extrema_window_start[None, :]
    #         )

    #         # An extremum is detected if any event overlaps its window.
    #         detected_extrema = np.any(overlap_matrix, axis=0)

    #         # An event is considered matched if it overlaps at least one
    #         # expected extremum window.
    #         matched_events = np.any(overlap_matrix, axis=1)

    #         n_detected = int(np.sum(detected_extrema))
    #         n_matched_events = int(np.sum(matched_events))
    #         n_spurious_events = int(len(events) - n_matched_events)

    #         completeness = n_detected / n_expected
    #         purity = n_matched_events / len(events)

    #         self.injections.loc[
    #             inj_idx,
    #             "n_detected_extrema",
    #         ] = n_detected

    #         self.injections.loc[
    #             inj_idx,
    #             "n_matched_events",
    #         ] = n_matched_events

    #         self.injections.loc[
    #             inj_idx,
    #             "n_spurious_events",
    #         ] = n_spurious_events

    #         self.injections.loc[
    #             inj_idx,
    #             "var_completeness",
    #         ] = completeness

    #         self.injections.loc[
    #             inj_idx,
    #             "var_purity",
    #         ] = purity

    #         # --------------------------------------------------------------
    #         # Phase RMS
    #         #
    #         # For each matched event, compare its peak time with the nearest
    #         # expected extremum. Returned as a fraction of half a cycle.
    #         # --------------------------------------------------------------
    #         matched = events.loc[matched_events]

    #         if len(matched) > 0:

    #             peak_times = matched.mjd_max.to_numpy()

    #             nearest_extremum_offsets = np.min(
    #                 np.abs(
    #                     peak_times[:, None]
    #                     - extrema_times[None, :]
    #                 ),
    #                 axis=1,
    #             )

    #             phase_offsets = (
    #                 nearest_extremum_offsets
    #                 / half_period_days
    #             )

    #             phase_rms = np.sqrt(
    #                 np.mean(phase_offsets ** 2)
    #             )

    #             self.injections.loc[
    #                 inj_idx,
    #                 "var_phase_rms",
    #             ] = phase_rms

    #         # --------------------------------------------------------------
    #         # Largest run of consecutive missed extrema
    #         # --------------------------------------------------------------
    #         missed = ~detected_extrema

    #         largest_gap = 0
    #         current_gap = 0

    #         for is_missed in missed:
    #             if is_missed:
    #                 current_gap += 1
    #                 largest_gap = max(largest_gap, current_gap)
    #             else:
    #                 current_gap = 0

    #         self.injections.loc[
    #             inj_idx,
    #             "largest_extrema_gap",
    #         ] = largest_gap

    #         self.injections.loc[
    #             inj_idx,
    #             "largest_extrema_gap_min",
    #         ] = (
    #             largest_gap
    #             * half_period_days
    #             * 1440.0
    #         )

    #         self.injections.loc[inj_idx, "detected"] = (
    #             "y" if n_detected > 0 else "n"
    #         )

    #     return self.injections


        

    def gather_results(self,cut,centroid_match_radius=1.0, 
                       min_temporal_iou=0.0, spatial_weight=0.5,overlap_weight=2.0,
                       duration_weight=0.5,peak_weight=0.1):

        if cut != self.cut:
            self.nav.gather_data(cut=cut)
            self.nav.gather_results(cut=cut,isolated=True)
            self.cut = cut

        directory = f'{self.path}/Cut{cut}of{self.n**2}'

        self.injections = pd.read_csv(f'{directory}/{self.injection_dir}/injected_events.csv')

        # TEMPORARY #
        try:
            self.injections['mjd_max'] = self.nav.time[self.injections['frame_max'].values]
            self.injections.reset_index(names='injid',inplace=True)
        except:
            pass
        # TEMPORARY #

        self.transients = self.match_results_to_transients(centroid_match_radius, 
                                                           min_temporal_iou, spatial_weight,
                                                           overlap_weight,duration_weight,
                                                           peak_weight)
        # self.match_vars_to_injections()

    def filter_transients(self,cut=None,min_frame_duration=None,max_frame_duration=None,
                          min_snr=None,max_snr=None,min_K=None,max_K=None,event_type=None,detected=None):

        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!') 

        transients = self.transients.copy()

        if event_type is not None:
            transients = transients[transients.event_type == event_type]
        if detected is not None:
            transients = transients[transients.detected == detected]     
        if min_frame_duration is not None:
            transients = transients[transients.frame_duration >= min_frame_duration]
        if max_frame_duration is not None:
            transients = transients[transients.frame_duration <= max_frame_duration]
        if min_snr is not None:
            transients = transients[transients.snr >= min_snr]
        if max_snr is not None:
            transients = transients[transients.snr <= max_snr]
        if min_K is not None:
            transients = transients[transients.K <= min_K]
        if max_K is not None:
            transients = transients[transients.K <= max_K]

        return transients
            

    def plot_lc(self,injid,cut=None,frame_buffer=10):

        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')

        inj = self.transients[self.transients.injid==injid].iloc[0]
        lc = np.load(f'{self.path}/Cut{cut}of{self.n**2}/{self.injection_dir}/lightcurves.npz',allow_pickle=True)['lcs'][injid]

        plt.figure()
        plt.plot(self.nav.time[lc[0].astype(int)],lc[1],'d-',c='r',label='Injected Flux')

        if inj.ev_match == '-':
            xint = RoundToInt(inj.xcentroid)
            yint = RoundToInt(inj.ycentroid)
        else:
            objid,eventid = np.array(inj.ev_match.split('_')).astype(int)
            match_ev = self.nav.events[(self.nav.events.objid==objid)&(self.nav.events.eventid==eventid)].iloc[0]
            
            cadence = np.nanmedian(np.diff(self.nav.time)) * match_ev.frame_bin
            xint = int(match_ev.xint)
            yint = int(match_ev.yint)
            plt.axvspan(match_ev.mjd_start-cadence/2, match_ev.mjd_end+cadence/2, color='C1', alpha=0.4)

        cube_lc = np.nansum(self.nav.flux[:,yint-1:yint+2,xint-1:xint+2],axis=(1,2))

        plt.plot(self.nav.time,cube_lc,'x-',c='k',label='Cube Flux')

        xmin = self.nav.time[(lc[0][0]-frame_buffer).astype(int)]
        xmax = self.nav.time[(lc[0][-1]+frame_buffer).astype(int)]

        visible = (self.nav.time >= xmin) & (self.nav.time <= xmax)
        y_visible = cube_lc[visible]
        padding = 0.05 * (np.nanmax(y_visible) - np.nanmin(y_visible))

        plt.xlim(xmin,xmax)
        plt.ylim(np.nanmin(y_visible) - padding,np.nanmax(y_visible) + padding)

        plt.legend()
        plt.xlabel('Time [MJD]')
        plt.ylabel('TESS Counts')

    def plot_frames(self,injid,cut=None,
                    sources=False,isolated=False,events=True,
                    image_size=11,vmin=10,vmax=90):
        """
        Extract cutout images for chosen event.
        """

        import matplotlib.gridspec as gridspec

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')

        # -- Isolate event -- #
        inj = self.transients[self.transients.injid==injid].iloc[0]

        # -- Define cutout -- #
        xint = RoundToInt(inj.xcentroid)
        yint = RoundToInt(inj.ycentroid)
        brightest_frame = RoundToInt(inj.frame_max)
        xmin = max(xint-image_size//2,0)
        xmax = min(xint+image_size//2,self.nav.flux.shape[1])
        ymin = max(yint-image_size//2,0)
        ymax = min(yint+image_size//2,self.nav.flux.shape[1])

        brightest_im = self.nav.flux[brightest_frame,ymin:ymax+1,xmin:xmax+1]

        vmax = np.percentile(brightest_im[image_size//2-1:image_size//2+2, image_size//2-1:image_size//2+2], vmax)
        vmin = np.percentile(brightest_im[image_size//2-1:image_size//2+2, image_size//2-1:image_size//2+2], vmin)

        frames = np.arange(brightest_frame-2, brightest_frame+3).astype(int)

        fig = plt.figure(figsize=(15, 3))
        gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.35)
        ax = [fig.add_subplot(gs[i]) for i in range(5)]
        cax = fig.add_subplot(gs[5])

        for i,frame in enumerate(frames):
            im = ax[i].imshow(self.nav.flux[frame], origin='lower', cmap='gray', vmax=vmax, vmin=vmin)
            ax[i].axis('off')

            if sources:
                frame_sources = self.nav.sources[(self.nav.sources.frame_bin==1) & (self.nav.sources.frame==frame)]
                ax[i].scatter(frame_sources.xcentroid,frame_sources.ycentroid,c='orange',s=3,label='Sources')

            if isolated:
                frame_isosources = self.nav.isolated[self.nav.isolated.frame==frame]
                ax[i].scatter(frame_isosources.xcentroid,frame_isosources.ycentroid,c='r',s=3,label='Iso Sources')

            if events:
                frame_events = self.nav.events[(self.nav.events.frame_bin==1) 
                                               & (self.nav.events.frame_start <= frame)
                                               & (self.nav.events.frame_end >= frame)]
                ax[i].scatter(frame_events.xcentroid,frame_events.ycentroid,c='green',s=3,label='Events')

            ax[i].set_xlim(xint-image_size//2,xint+image_size//2)
            ax[i].set_ylim(yint-image_size//2,yint+image_size//2)

            if i == 0:
                ax[i].legend()

            if i == 2:
                ax[i].set_title(f'Brightest Frame ({frame})')
            else:
                ax[i].set_title(f'Frame {frame}')

        fig.colorbar(im, cax=cax, label='TESS Counts')

        # Snap colorbar to exactly match ax[4]'s height and sit close beside it
        fig.canvas.draw()
        pos = ax[4].get_position()
        cax.set_position([pos.x1 + 0.01, pos.y0, 0.01, pos.height])

        # if not plot:
        #     plt.close()
        
        # if return_plot:
        #     return images,fig
        # else:
        #     return images
        

    def compare_columns(self,columns,cut=None,log_columns=[]):

        # -- Gather data -- #
        if cut is None:
            cut = self.cut
        if cut is None:
            raise ValueError('Please specify a cut!')

        plot_df = self.transients[columns].copy()
        for c in log_columns:
            plot_df[f'log_{c}'] = np.log10(plot_df[c])  # rename or relabel axes after
            plot_df.drop(columns=c,inplace=True)

        sns.pairplot(plot_df, corner=True, diag_kind="kde")


