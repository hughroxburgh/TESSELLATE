import numpy as np
import os
import pandas as pd

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def save_compact_array(path,arr):
    """
    Save an ndarray with np.save, downcasting float64 to float32 first.
    Reduces on-disk size for large reduction products (flux/background cubes)
    with no precision loss that matters for photometry. Non-float64 arrays
    (bool masks, int arrays, existing float32) pass through unchanged.
    """
    arr = np.asarray(arr)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    np.save(path,arr)

def save_table(df,path):
    """
    Save a DataFrame as Parquet instead of CSV to reduce on-disk size.
    `path` should be the '.csv' path used historically by callers; the file
    is actually written alongside it with a '.parquet' extension.
    """
    parquet_path = path[:-4]+'.parquet' if path.endswith('.csv') else path+'.parquet'
    df.to_parquet(parquet_path,index=False)

def load_table(path):
    """
    Load a DataFrame saved with save_table. Looks for the Parquet file first,
    falling back to the legacy '.csv' path so pre-existing outputs on disk
    remain readable without needing to be regenerated.
    """
    parquet_path = path[:-4]+'.parquet' if path.endswith('.csv') else path
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.read_csv(path)

def table_exists(path):
    """
    True if a table saved with save_table exists, checking both the
    '.parquet' path and the legacy '.csv' path.
    """
    parquet_path = path[:-4]+'.parquet' if path.endswith('.csv') else path
    return os.path.exists(parquet_path) or os.path.exists(path)

def _Save_space(Save,delete=False):
    """
    Creates a path if it doesn't already exist.
    """
    try:
        os.makedirs(Save)
    except FileExistsError:
        if delete:
            os.system(f'rm -r {Save}/')
            os.makedirs(Save)
        else:
            pass

def _Remove_emptys(files):
    """
    Deletes corrupt fits files before creating cube
    """

    deleted = 0
    for file in files:
        size = os.stat(file)[6]
        if size < 35500000:
            os.system('rm ' + file)
            deleted += 1
    return deleted

def _Extract_fits(pixelfile):
    """
    Quickly extract fits
    """
    from astropy.io import fits

    try:
        hdu = fits.open(pixelfile)
        return hdu
    except OSError:
        print('OSError ',pixelfile)
        return
    
def _Print_buff(length,string):

    strLength = len(string)
    buff = '-' * int((length-strLength)/2)
    return f"{buff}{string}{buff}"

def _Check_dirs(save_path):
    """
    Check that all reduction directories are constructed.

    Parameters:
    -----------
    dirlist : list
        List of directories to check for, if they don't exist, they will be created.
    """
    import os
    #for d in dirlist:
    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except:
            pass

def _Submit_sbatch(script_path, retries=5, delay=3):
    """Submit a SLURM batch script via sbatch, retrying on transient submission
    failures instead of silently dropping the job or crashing on an empty/
    malformed sbatch response -- at the scale of tens of thousands of
    submissions across many sectors, an occasional busy-slurmctld hiccup is
    expected and shouldn't take down the whole driver run. Returns the job id
    as a string, or raises RuntimeError if every attempt fails.
    """
    import subprocess
    from time import sleep as _sleep

    last_err = None
    for attempt in range(retries):
        result = subprocess.run(f'sbatch {script_path}', shell=True, capture_output=True, text=True)
        parts = result.stdout.strip().split()
        if parts and parts[-1].isdigit():
            return parts[-1]
        last_err = result.stderr.strip() or result.stdout.strip() or '(no output)'
        _sleep(delay)
    raise RuntimeError(f'sbatch failed after {retries} attempts for {script_path}: {last_err}')


def _Check_job_status(job_id):
    """
    Returns one of: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, UNKNOWN
    Checks squeue first (job is still active), then sacct (job has finished).
    """

    import subprocess
    from time import sleep as _sleep

    # squeue only shows active/queued jobs
    sq = subprocess.run(
        f'squeue -j {job_id} -h -o %T',
        shell=True, capture_output=True, text=True
    )
    state = sq.stdout.strip()

    if state:  # Job is still in the queue
        return state  # e.g. PENDING, RUNNING, COMPLETING

    # Job has left the queue -- check sacct for final state. A job that just left
    # squeue can briefly be invisible to sacct too (accounting propagation lag),
    # which would otherwise look identical to a genuinely unknown/bad job id and
    # crash the caller -- retry a few times before conceding UNKNOWN.
    for attempt in range(5):
        sa = subprocess.run(
            f'sacct -j {job_id} -o State -n -X',
            shell=True, capture_output=True, text=True
        )
        if sa.stdout.strip():
            return sa.stdout.strip().split()[0]
        _sleep(3)
    return 'UNKNOWN'


def _remove_ffis(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/image_files')

    os.chdir(home_path)

def _remove_cubes(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            if part:
                for i in range(1,3):
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}')
                        os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_cube.fits')
                        os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_wcs.fits')
                        os.system(f'rm -f cubed.txt')
                    except:
                        pass
            else:
                try:
                    os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}')
                    os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_cube.fits')
                    os.system(f'rm -f sector{sector}_cam{cam}_ccd{ccd}_wcs.fits')
                    os.system(f'rm -f cubed.txt')
                except:
                    pass

    os.chdir(home_path)

def _remove_cuts(data_path,sector,n,cams,ccds,cuts,part):

    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                else:
                    os.system(f'rm -r -f {data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')

def _remove_asteroids(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f asteroids/*_Asteroids.parquet')
                            os.system(f'rm -f asteroids/*_AsteroidTrails.png')
                            os.system(f'rm -f asteroids.txt')
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f asteroids/*_Asteroids.parquet')
                        os.system(f'rm -f asteroids/*_AsteroidTrails.png')
                        os.system(f'rm -f asteroids.txt')
                    except:
                        pass

    os.chdir(home_path)

def _remove_reductions(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f *.npy')
                            os.system(f'rm -f reduced.txt')
                            os.system(f'rm -f detected_events.csv')
                            os.system(f'rm -f detected_sources.csv')
                            os.system(f'rm -f detected_objects.csv')
                            os.system('rm -f figs.zip')
                            os.system('rm -f lcs.zip')  
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f *.npy')
                        os.system(f'rm -f reduced.txt')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system(f'rm -f detected_objects.csv')
                        os.system('rm -f figs.zip')
                        os.system('rm -f lcs.zip')  
                    except:
                        pass   

    os.chdir(home_path)

def _remove_asteroid_lightcurves(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f asteroids/*_AsteroidAperturePhotometry.parquet')
                            os.system(f'rm -f asteroids/*_AsteroidPSFPhotometry.parquet')
                            os.system(f'rm -f asteroids/*_AsteroidStackSummary.parquet')
                            os.system(f'rm -f asteroids/*_AsteroidStackedPhotometry.parquet')
                            os.system(f'rm -f asteroids/*_AsteroidCutOffset.parquet')
                            os.system(f'rm -f asteroid_lightcurves.txt')
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f asteroids/*_AsteroidAperturePhotometry.parquet')
                        os.system(f'rm -f asteroids/*_AsteroidPSFPhotometry.parquet')
                        os.system(f'rm -f asteroids/*_AsteroidStackSummary.parquet')
                        os.system(f'rm -f asteroids/*_AsteroidStackedPhotometry.parquet')
                        os.system(f'rm -f asteroids/*_AsteroidCutOffset.parquet')
                        os.system(f'rm -f asteroid_lightcurves.txt')
                    except:
                        pass

    os.chdir(home_path)

def _remove_calibrations(data_path, sector, n, cams, ccds, cuts, part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                try:
                    os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                    # Calibration products now live in the calibration/ subdir
                    os.system('rm -rf calibration')
                    # Legacy: products previously written directly in the cut dir
                    os.system('rm -f calibrated.txt')
                    os.system('rm -f psf_calibration_*.csv')
                    os.system('rm -f psf_calibration_*.pdf')
                    os.system('rm -f detection_limits_*.csv')
                except:
                    pass
    os.chdir(home_path)

def _remove_search(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system(f'rm -f detected_events.csv')
                            os.system(f'rm -f detected_sources.csv')
                            os.system(f'rm -f detected_objects.csv')
                            os.system('rm -f figs.zip')
                            os.system('rm -f lcs.zip') 
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system(f'rm -f detected_events.csv')
                        os.system(f'rm -f detected_sources.csv')
                        os.system(f'rm -f detected_objects.csv')
                        os.system('rm -f figs.zip')
                        os.system('rm -f lcs.zip')  
                    except:
                        pass 
    os.chdir(home_path)
    
def _remove_plots(data_path,sector,n,cams,ccds,cuts,part):

    home_path = os.getcwd()
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                if part:
                    for i in range(1,3):
                        try:
                            os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Part{i}/Cut{cut}of{n**2}')
                            os.system('rm -r -f figs.zip')
                            os.system('rm -r -f lcs.zip')
                        except:
                            pass
                else:
                    try:
                        os.chdir(f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{n**2}')
                        os.system('rm -r -f figs.zip')
                        os.system('rm -r -f lcs.zip')
                    except:
                        pass 
    os.chdir(home_path)

def delete_files(filetype,data_path,sector,n=4,cams='all',ccds='all',cuts='all',part=False):

    if cams == 'all':
        cams = [1,2,3,4]
    elif type(cams) == int:
        cams = [cams]
    if ccds == 'all':
        ccds = [1,2,3,4]
    elif type(ccds) == int:
        ccds = [ccds]
    if cuts == 'all':
        cuts = np.linspace(1,n**2,n**2).astype(int)
    elif type(cuts) == int:
        cuts = [cuts]
        
    possibleFiles = {'ffis':_remove_ffis,
                        'cubes':_remove_cubes,
                        'cuts':_remove_cuts,
                        'asteroids':_remove_asteroids,
                        'reductions':_remove_reductions,
                        'asteroid_lightcurves':_remove_asteroid_lightcurves,
                        'calibrations':_remove_calibrations,
                        'search':_remove_search,
                        'plot':_remove_plots}

    if filetype.lower() in possibleFiles.keys():
        function = possibleFiles[filetype.lower()]
        function(data_path,sector,n,cams,ccds,cuts,part)
    else:
        e = 'Invalid filetype! Valid types: "ffis" , "cubes" , "cuts" , "asteroids" , "reductions" , "asteroid_lightcurves" , "calibrations" , "search", "plot". '
        raise AttributeError(e)
    
    
def weighted_avg_var(group, weight_col):
    import pandas as pd

    weighted_stats = {}
    numeric_cols = group.select_dtypes(include='number').columns  # Select only numeric columns
    miss = ['Prob','n_detections','GaiaID','flux_sign',
            'ra_source','x_source','y_source','dec_source',
            'e_ra_source','e_x_source','e_y_source','e_dec_source',
            'source_mask','eventID','xccd_source','yccd_source',
            'e_xccd_source','e_yccd_source']
    if len(group) == 1:  # If group has only one entry, return the row but with the same column structure
        original_row = group.iloc[0].copy()
        result = {}
        for col in numeric_cols:
            if (col != 'objid') & (col not in miss):
                result[col] = original_row[col]  # Original value
                result[f'e_{col}'] = np.nan  # No variance
            else:
                result[col] = group[col].iloc[0]
        return pd.Series(result)
    
    for col in numeric_cols:
        if (col != 'objid') & (col not in miss):  # Exclude the weight column itself
            # Compute the weighted average using nansum
            weighted_avg = np.ma.masked_invalid(group[col] * group[weight_col]).sum() / np.ma.masked_invalid(group[weight_col]).sum()
            # Compute the weighted variance using nansum
            variance = np.ma.masked_invalid(group[weight_col] * (group[col] - weighted_avg) ** 2).sum() / np.ma.masked_invalid(group[weight_col]).sum()
            # Store both weighted average and variance
            weighted_stats[col] = weighted_avg
            weighted_stats[f'e_{col}'] = variance
        else:
            weighted_stats[col] = group[col].iloc[0]
    return pd.Series(weighted_stats)


def pandas_weighted_avg(df,weight_col='sig'):
    # df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col,include_groups=False).reset_index()
    # print(df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col).head())
    df = df.groupby('objid').apply(weighted_avg_var, weight_col=weight_col).reset_index(drop=True)
    return df

def consecutive_points(data, stepsize=2):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def Gaussian(t, A, t0, sigma, offset):
    return A * np.exp(-0.5 * ((t - t0) / sigma)**2) + offset

def Distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def RoundToInt(num):
    return np.floor(num+0.5).astype(int)

def Exp_func(x,a,b,c):
   e = np.exp(a)*np.exp(-x/np.exp(b)) + np.exp(c)
   return e






def _orbit_ref_correction(lc_flux, time, full_time, orbit_refs, orbit_segments, x, y, radius=1.5):
    """Correct inter-orbit flux offset caused by per-orbit reference subtraction.

    Each orbit's reference image is measured through the same aperture as the
    science LC. The orbit with the lowest-noise reference is the primary; all
    other orbits have the delta (primary_ref_flux - orbit_ref_flux) added so
    that every orbit is on the same flux baseline.

    Parameters
    ----------
    lc_flux       : array (N,)  — LC flux values (may be a subset of full time series)
    time          : array (N,)  — times corresponding to lc_flux
    full_time     : array (T,)  — full time array matching orbit_segments
    orbit_refs    : dict {int: ndarray (NY, NX)}
    orbit_segments: ndarray (T,)
    x, y          : float — source position (col, row)
    radius        : float

    Returns
    -------
    corrected lc_flux : array (N,)
    """
    xint = int(np.round(x, 0))
    yint = int(np.round(y, 0))
    buf = int(np.floor(radius))

    # Measure aperture flux on each orbit reference image
    orb_flux = {}
    orb_std = {}
    for seg, ref_im in orbit_refs.items():
        stamp = ref_im[yint-buf:yint+buf+1, xint-buf:xint+buf+1]
        orb_flux[seg] = np.nansum(stamp)
        orb_std[seg] = np.nanstd(ref_im)

    primary = min(orb_std, key=orb_std.get)
    f_primary = orb_flux[primary]

    corrected = lc_flux.copy()
    for seg in orbit_refs:
        if seg == primary:
            continue
        delta = f_primary - orb_flux[seg]
        # Find which indices in lc_flux belong to this orbit
        full_mask = orbit_segments == seg
        seg_times = set(full_time[full_mask])
        lc_mask = np.array([t in seg_times for t in time])
        corrected[lc_mask] += delta

    return corrected


def Generate_LC(time,flux,x,y,frame_start=None,frame_end=None,method='sum',radius=1.5):#,
                #orbit_refs=None,orbit_segments=None):

    from photutils.aperture import CircularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
    from scipy.signal import fftconvolve

    full_time = time
    t = time
    f = flux

    if frame_start is not None:
        if frame_end is not None:
            t = t[frame_start:frame_end+1]
            f = f[frame_start:frame_end+1]
        else:
            t = t[frame_start:]
            f = f[frame_start:]
    elif frame_end is not None:
        t = t[:frame_end+1]
        f = f[:frame_end+1]

    if method.lower() == 'aperture':
        aperture = CircularAperture([x, y], radius)
        annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
        flux = []
        flux_err = []
        for i in range(len(f)):
            m = sigma_clip(data,masked=True,sigma=5).mask
            mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
            aperstats_sky = ApertureStats(f[i], annulus_aperture,mask = mask)
            phot_table = aperture_photometry(f[i], aperture)
            bkg_std = aperstats_sky.std
            flux_err += [aperture.area * bkg_std]
            flux += [phot_table['aperture_sum'].value[0]]
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        # if orbit_refs is not None and orbit_segments is not None:
        #     flux = _orbit_ref_correction(flux, t, full_time, orbit_refs, orbit_segments, x, y, radius)
        return t, flux, flux_err
    elif method.lower() == 'sum':
        xint = int(np.round(x,0))
        yint = int(np.round(y,0))
        buffer = np.floor(radius).astype(int)
        lc = np.nansum(f[:,yint-buffer:yint+buffer+1,xint-buffer:xint+buffer+1],axis=(1,2))
        # if orbit_refs is not None and orbit_segments is not None:
        #     lc = _orbit_ref_correction(lc, t, full_time, orbit_refs, orbit_segments, x, y, radius)
        return t, lc



def Get_Tess_Vectors(sector, camera, data_path='/fred/oz335/_local_TESS_vectors'):
    """Return a TESSVectors DataFrame for *sector* / *camera*."""

    import pandas as pd

    _TESSVECTORS_FNAME = "TessVectors_S{sector:03d}_C{camera}_FFI.csv"
    fname = _TESSVECTORS_FNAME.format(sector=sector, camera=camera)

    local = os.path.join(data_path, fname)
    if os.path.isfile(local):
        return pd.read_csv(local, comment='#', index_col=False)

def Get_Tess_Downlink(sector, camera, time):

    df = Get_Tess_Vectors(sector, camera)

    btjd = time - 56999.5
    vec_t = df['MidTime'].values
    segment = np.interp(btjd, vec_t, df['Segment'].values).astype(int)
    break_idx = np.argmax(np.diff(segment)) + 1

    return break_idx

def Frame_Bin(sector, camera, time, flux=None, frame_bin=1):

    break_idx = Get_Tess_Downlink(sector, camera, time)

    def _bin_segment(t, f=None):
        points = np.arange(0, len(t), frame_bin)
        binned_time = np.array([np.nanmean(t[i:i+frame_bin]) for i in points])
        if f is None:
            return binned_time
        binned_flux = np.array([np.nanmean(f[i:i+frame_bin], axis=0) for i in points])
        return binned_time, binned_flux

    if flux is None:
        return np.concatenate([_bin_segment(time[:break_idx]), _bin_segment(time[break_idx:])])
    
    time_a, flux_a = _bin_segment(time[:break_idx], flux[:break_idx])
    time_b, flux_b = _bin_segment(time[break_idx:], flux[break_idx:])
    return np.concatenate([time_a, time_b]), np.concatenate([flux_a, flux_b])



def manual_sort(events_path, image_dir=None, sort_dir=None):
    """
    Sort events by eye into categories named when it starts (e.g. 'Real' and
    'Not real'), one keypress per event, in an OpenCV window like
    development/manual_sort.py.

    events_path : csv of events (sector, camera, ccd, cut, objid, eventid, ...).
    image_dir : folder searched, subfolders included, for the event images
        S{sector}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png made by plot_lc
        (e.g. development/plot_events.py). Default: 'images' next to the csv.
    sort_dir : where the sorted copies go. Default: 'sort_{csv name}' next to the csv.

    Images are copied, not moved, so the same images can be sorted in several
    ways. Each category gets a folder of images plus an events.csv of their
    rows -- the layout ml_classifier.load_manual_labels reads. Running again
    with the same sort_dir carries on where it stopped.

    Keys: 1-9 sort, Enter skip, Backspace undo, Esc or q quit.
    """

    import cv2
    import shutil

    keys = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
    base = os.path.dirname(os.path.abspath(events_path))
    name = os.path.splitext(os.path.basename(events_path))[0]
    image_dir = os.path.abspath(image_dir or os.path.join(base, 'images'))
    sort_dir = os.path.abspath(sort_dir or os.path.join(base, f'sort_{name}'))
    order_file = os.path.join(sort_dir, 'categories.txt')

    events = pd.read_csv(events_path)

    # -- Categories: carry on an existing sort, or ask for new ones -- #
    classes = []
    if os.path.exists(order_file):
        with open(order_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        answer = input(f'Continue the sort in {sort_dir} with categories {classes}? [y/n] ')
        if answer.strip().lower() != 'y':
            print('Pass a different sort_dir to start a new sort.')
            return
    else:
        print('Enter category names one at a time (up to 9); press Enter on an empty line to finish.')
        while len(classes) < 9:
            c = input(f'  Category {len(classes) + 1}: ').strip()
            if not c:
                break
            if c in classes or '/' in c or '\\' in c:
                print('  Skipped: duplicate name or contains a slash.')
                continue
            classes.append(c)
        if len(classes) < 2:
            print('Need at least two categories.')
            return
        os.makedirs(sort_dir, exist_ok=True)
        with open(order_file, 'w') as f:
            f.write('\n'.join(classes) + '\n')

    group_csv = {c: os.path.join(sort_dir, c, 'events.csv') for c in classes}
    groups = {}
    for c in classes:
        os.makedirs(os.path.join(sort_dir, c), exist_ok=True)
        groups[c] = pd.read_csv(group_csv[c]) if os.path.exists(group_csv[c]) else events.iloc[:0].copy()

    # -- Find each event's image, and what is already sorted -- #
    image_of = {}
    for root, _, files in os.walk(image_dir):
        if os.path.abspath(root).startswith(sort_dir):
            continue
        for f in files:
            if f.endswith('.png'):
                image_of.setdefault(f, os.path.join(root, f))
    done = {f for c in classes for f in os.listdir(os.path.join(sort_dir, c)) if f.endswith('.png')}

    names = [f'S{r[0]}C{r[1]}C{r[2]}C{r[3]}O{r[4]}E{r[5]}.png' for r in events[keys].itertuples(index=False)]
    todo = [(i, n) for i, n in enumerate(names) if n not in done and n in image_of]
    n_missing = sum(n not in image_of for n in names)
    print(f'{len(events)} events: {sum(n in done for n in names)} already sorted, {len(todo)} to sort, '
          f'{n_missing} without an image in {image_dir}')
    if not todo:
        return

    # -- Windows -- #
    screen_w, screen_h = 1920, 1080
    try:
        from screeninfo import get_monitors
        screen_w, screen_h = get_monitors()[0].width, get_monitors()[0].height
    except Exception:
        pass

    lines = [f'{k + 1} : {c}' for k, c in enumerate(classes)] + ['', 'Enter : skip', 'Bkspc : undo', 'Esc/q : quit']
    guide = np.zeros((32 * len(lines) + 20, 380, 3), dtype=np.uint8)
    guide[:] = (0, 0, 139)
    for k, line in enumerate(lines):
        cv2.putText(guide, line, (15, 35 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.imshow('Controls', guide)
    cv2.moveWindow('Controls', 0, max(screen_h - guide.shape[0] - 60, 0))
    cv2.waitKey(1)

    # -- Sorting loop -- #
    history = []    # (position in todo, category or None if skipped)
    i = 0
    while i < len(todo):
        idx, fname = todo[i]
        img = cv2.imread(image_of[fname])
        if img is None:
            print(f'Could not read {fname}, skipping.')
            i += 1
            continue
        if img.shape[1] > 0.8 * screen_w:
            scale = 0.8 * screen_w / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        cv2.imshow('Image Sorter', img)
        cv2.moveWindow('Image Sorter', 0, 0)
        print(f'[{i + 1}/{len(todo)}] {fname}')

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break

        if key in (8, 127):
            if not history:
                print('  Nothing to undo.')
                continue
            j, c = history.pop()
            if c is not None:
                prev = todo[j][1]
                copied = os.path.join(sort_dir, c, prev)
                if os.path.exists(copied):
                    os.remove(copied)
                ev = events.iloc[todo[j][0]]
                match = np.all([groups[c][k].to_numpy() == ev[k] for k in keys], axis=0)
                groups[c] = groups[c][~match]
                groups[c].to_csv(group_csv[c], index=False)
                print(f'  Undid {prev} (was {c})')
            i = j
            continue

        if key == 13:
            history.append((i, None))
            print('  Skipped')
            i += 1
            continue

        if ord('1') <= key < ord('1') + len(classes):
            c = classes[key - ord('1')]
            shutil.copy2(image_of[fname], os.path.join(sort_dir, c, fname))
            groups[c] = pd.concat([groups[c], events.iloc[[idx]]], ignore_index=True)
            groups[c].to_csv(group_csv[c], index=False)
            history.append((i, c))

            overlay = img.copy()
            (tw, th), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            cx, cy = (overlay.shape[1] - tw) // 2, (overlay.shape[0] + th) // 2
            cv2.rectangle(overlay, (cx - 10, cy - th - 10), (cx + tw + 10, cy + 10), (0, 0, 255), -1)
            cv2.putText(overlay, c, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Image Sorter', overlay)
            cv2.waitKey(300)
            print(f'  -> {c}')
            i += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    counts = {c: sum(f.endswith('.png') for f in os.listdir(os.path.join(sort_dir, c))) for c in classes}
    print('Sorted so far: ' + ', '.join(f'{c} {n}' for c, n in counts.items()))
    print(f'Results in {sort_dir}')
