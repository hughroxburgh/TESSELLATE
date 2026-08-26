import numpy as np
import pandas as pd
from time import time as clock
from copy import deepcopy
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from .tools import RoundToInt, load_table, table_exists


# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- # 
# ------------------------------------------------- Source Detection functions ------------------------------------------------ #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #


# TESSSourceFinder Funcs (from Claude)

def _Mexhat_kernels(fwhm: float) -> np.ndarray:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    size = max(5, int(np.ceil(6 * sigma)) | 1)
    c = size // 2
    y, x = np.mgrid[-c:c+1, -c:c+1]
    r2 = (x**2 + y**2) / (2 * sigma**2)
    k = (1 - r2) * np.exp(-r2)
    return (k / np.abs(k).sum()).astype(np.float32)


def _Dedup_close_sources(combined: pd.DataFrame, dedup_radius: float) -> pd.DataFrame:
    from scipy.spatial import KDTree
    keep = np.ones(len(combined), dtype=bool)
    tree = KDTree(combined[["xcentroid", "ycentroid"]].values)
    for i in range(len(combined)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(combined[["xcentroid", "ycentroid"]].iloc[i].values, r=dedup_radius):
            if j > i:
                keep[j] = False
    return combined[keep].reset_index(drop=True)


# def _Negative_pixel_extent(data_sub: np.ndarray, sources: pd.DataFrame, r: float = 2.0) -> np.ndarray:
#     from astropy.stats import sigma_clipped_stats
#     _, _, std = sigma_clipped_stats(data_sub)
#     H, W = data_sub.shape
#     R = int(np.ceil(r))
#     dy, dx = np.mgrid[-R:R+1, -R:R+1]
#     in_circle = (dx**2 + dy**2) <= r**2
#     dy_offsets = dy[in_circle]
#     dx_offsets = dx[in_circle]

#     xs = np.round(sources["xcentroid"].values).astype(int)
#     ys = np.round(sources["ycentroid"].values).astype(int)

#     extents = np.empty(len(sources), dtype=float)
#     for i, (cx, cy) in enumerate(zip(xs, ys)):
#         px = cx + dx_offsets
#         py = cy + dy_offsets
#         valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
#         if valid.sum() == 0:
#             extents[i] = np.nan
#             continue
#         extents[i] = max(0.0, -float(data_sub[py[valid], px[valid]].min())) / std

#     return extents

def _Negative_pixel_extent(data_sub, sources, r=2.0):
    from astropy.stats import sigma_clipped_stats
    _, _, std = sigma_clipped_stats(data_sub)
    H, W = data_sub.shape
    R = int(np.ceil(r))
    dy, dx = np.mgrid[-R:R+1, -R:R+1]
    in_circle = (dx**2 + dy**2) <= r**2
    dy_off = dy[in_circle]
    dx_off = dx[in_circle]

    xs = np.round(sources["xcentroid"].values).astype(int)
    ys = np.round(sources["ycentroid"].values).astype(int)

    # Shape: (n_sources, n_offsets)
    px = xs[:, None] + dx_off[None, :]
    py = ys[:, None] + dy_off[None, :]

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    patch = data_sub[py, px]  # (n_sources, n_offsets)
    patch = np.where(valid, patch, 0.0)
    min_vals = patch.min(axis=1)

    return np.maximum(0.0, -min_vals) / std

def _TESS_sourcefinder(image, frame_number, thresh = 0.3, bw=24,fwhm_min=0.7,fwhm_max=2.0, n_scales=10,deblend_nthresh=128,snr=3.0,
                       deblend_cont= 5e-5,dedup_radius= 1.0,flux_signs=(-1,1)):#,boundary_buffer= 2.0,boundary_near= 5.0):

    import sep

    fwhms = np.linspace(fwhm_min, fwhm_max, n_scales)
    kernels = [_Mexhat_kernels(f) for f in fwhms]

    allsources = pd.DataFrame()
    for flux_sign in flux_signs:
        data = np.asarray(image*flux_sign, dtype=np.float32)
        bkg = sep.Background(data, bw=bw, bh=bw)
        data_sub = data - bkg.back()
        err = bkg.rms()

        per_scale = []
        for fwhm, kern in zip(fwhms, kernels):
            try:
                objs = sep.extract(
                    data_sub, thresh=thresh, err=err, minarea=2,
                    deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                    filter_kernel=kern,
                )
            except Exception:
                continue
            if len(objs) == 0:
                continue
            per_scale.append(pd.DataFrame({
                "xcentroid": objs["x"], "ycentroid": objs["y"],
                "flux": objs["flux"], "peak": objs["peak"],
                "a": objs["a"], "b": objs["b"],
                "theta": objs["theta"], "flag": objs["flag"],
                "detection_fwhm": fwhm,
            }))

        empty = pd.DataFrame(columns=["xcentroid","ycentroid","flux","peak","a","b","theta","flag",
                                    "detection_fwhm","snr","ellipticity","fwhm",
                                    "neg_extent"])
        if not per_scale:
            continue

        sources = _Dedup_close_sources(
            pd.concat(per_scale, ignore_index=True)
            .sort_values("flux", ascending=False)
            .reset_index(drop=True),
            dedup_radius,
        )

        # H, W = data_sub.shape
        # x, y = sources["x"].values, sources["y"].values
        # dist_to_edge = np.minimum.reduce([x, y, (W - 1) - x, (H - 1) - y])
        # sources = sources[dist_to_edge >= boundary_buffer].reset_index(drop=True)

        # x, y = sources["x"].values, sources["y"].values
        # dist_to_edge = np.minimum.reduce([x, y, (W - 1) - x, (H - 1) - y])
        # sources["boundary_flag"] = dist_to_edge < boundary_near

        ap_flux, ap_err, _ = sep.sum_circle(
            data_sub, sources["xcentroid"].values, sources["ycentroid"].values,
            r=1.5, err=err, gain=1.0,
        )
        sources["snr"] = ap_flux / np.where(ap_err > 0, ap_err, np.nan)

        a = sources["a"].values.clip(min=1e-6)
        sources["ellipticity"] = 1.0 - sources["b"].values / a
        sources["fwhm"] = 2.0 * np.sqrt(np.log(2) * 2) * np.sqrt(sources["a"] * sources["b"])
        sources["neg_extent"] = _Negative_pixel_extent(data_sub, sources, r=2.0)
        sources['flux_sign'] = flux_sign
        sources['frame'] = frame_number

            # self.background_ = bkg
        allsources = pd.concat([allsources,sources[(sources.snr >= snr) & (sources.neg_extent < 3) & (sources.ellipticity<0.75)]])

    return allsources


def _Spatial_group(result,colname='objid',min_samples=1,distance=0.5,njobs=-1):
    """
    Groups events based on proximity.
    """

    from sklearn.cluster import DBSCAN

    pos = np.array([result.xcentroid,result.ycentroid]).T
    cluster = DBSCAN(eps=distance,min_samples=min_samples,n_jobs=njobs).fit(pos)
    labels = cluster.labels_
    unique_labels = set(labels)
    for label in unique_labels:
        result.loc[label == labels,colname] = label + 1
    result[colname] = result[colname].astype(int)
    return result

def _Spatio_temporal_group(result,colname='objid',min_samples=1,distance=0.5,frame_gap=5,njobs=-1):
    """
    Groups events based on proximity in both space and time, 
    then groups their median positions to identify repeating sources.
    This helps prevent moving items from creating large trailed artifact groups.
    """
    if len(result) == 0:
        return result

    from sklearn.cluster import DBSCAN

    # -- 1. Spatio-Temporal Clustering --
    # Scale frame distance so that a gap of `frame_gap` equals `distance` in DBSCAN 
    # Use max to avoid zero division if frame_gap is set to 0
    frame_scale = distance / max(1e-5, frame_gap)
    
    # Fallback to regular spatial group if no frame info is available
    if 'frame' not in result.columns:
        return _Spatial_group(result, colname=colname, min_samples=min_samples, distance=distance, njobs=njobs)
        
    scaled_frames = result['frame'].values * frame_scale
    
    pos_3d = np.array([result['xcentroid'].values, result['ycentroid'].values, scaled_frames]).T
    
    st_cluster = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=njobs).fit(pos_3d)
    st_labels = st_cluster.labels_.copy() # Make a copy to mutate
    
    # Handle noise (-1) from DBSCAN by giving them distinct unique labels 
    # so they can still participate in median clustering individually
    max_label = st_labels.max()
    noise_mask = st_labels == -1
    if noise_mask.sum() > 0:
        st_labels[noise_mask] = np.arange(max_label + 1, max_label + 1 + noise_mask.sum())
    
    result = result.copy()
    result['st_group'] = st_labels

    # -- 2. Cluster the median positions --
    medians = result.groupby('st_group')[['xcentroid', 'ycentroid']].median()
    
    pos_2d = medians[['xcentroid', 'ycentroid']].values
    spatial_cluster = DBSCAN(eps=distance, min_samples=1, n_jobs=njobs).fit(pos_2d)
    
    medians['final_label'] = spatial_cluster.labels_
    
    # Join labels back using st_group as the index mapping to medians index
    result = result.merge(medians[['final_label']], left_on='st_group', right_index=True, how='left')
    
    # Apply 1-indexed labeling identical to the original strategy
    result[colname] = (result['final_label'] + 1).fillna(0).astype(int)
    
    # Strip intermediate fields to maintain identical output structure
    result = result.drop(columns=['st_group', 'final_label'])
    
    return result

# # ---------- Star Finder source finding ---------- #
# def _Star_finding_procedure(data,prf,sig_limit = 2):
#     """
#     Use StarFinder to find stars with different PSF shapes depending on subpixel shift.
#     """

#     from photutils.detection import StarFinder
#     from astropy.stats import sigma_clipped_stats

#     mean, med, std = sigma_clipped_stats(data, sigma=5.0)

#     psfCentre = prf.locate(5,5,(11,11))
#     finder = StarFinder(med + sig_limit*std,kernel=psfCentre)
#     res1 = finder.find_stars(deepcopy(data))

#     psfUR = prf.locate(5.25,5.25,(11,11))
#     finder = StarFinder(med + sig_limit*std,kernel=psfUR)
#     res2 = finder.find_stars(deepcopy(data))

#     psfUL = prf.locate(4.75,5.25,(11,11))
#     finder = StarFinder(med + sig_limit*std,kernel=psfUL)
#     res3 = finder.find_stars(deepcopy(data))

#     psfDR = prf.locate(5.25,4.75,(11,11))
#     finder = StarFinder(med + sig_limit*std,kernel=psfDR)
#     res4 = finder.find_stars(deepcopy(data))

#     psfDL = prf.locate(4.75,4.75,(11,11))
#     finder = StarFinder(med + sig_limit*std,kernel=psfDL)
#     res5 = finder.find_stars(deepcopy(data))

#     tables = [res1, res2, res3, res4, res5]
#     good_tables = [table.to_pandas() for table in tables if table is not None]
#     if len(good_tables)>0:
#         total = pd.concat(good_tables)
#         total = total[~pd.isna(total['xcentroid'])]
#         if len(total) > 0:
#             grouped = _Spatial_group(total,distance=2)
#             res = grouped.groupby('objid').head(1)
#             res = res.reset_index(drop=True)
#             res = res.drop(['id','objid'],axis=1)
#         else:
#             res=None
#     else:
#         res = None

#     return res


# def _Find_stars(data,prf,fwhmlim=7,siglim=2.5,bkgstd_lim=50,negative=False):
#     """
#     Find stars in image.
#     """

#     from scipy.signal import fftconvolve
#     from photutils.aperture import RectangularAperture, RectangularAnnulus, ApertureStats, aperture_photometry
#     from astropy.stats import sigma_clip

#     # -- Look for negative detections by flipping sign of data -- #
#     if negative:
#         data = data * -1

#     # -- Use StarFinder to find sources with similar psf to TESS PRF -- #
#     star = _Star_finding_procedure(data,prf,sig_limit=siglim)

#     if star is None:
#         return None
    
#     # -- Initial cutoff, requiring fwhm to be star like and the source to be at least 3 pixels in from the edge -- #
#     ind = (star['fwhm'].values < fwhmlim) & (star['fwhm'].values > 0.8)
#     pos_ind = ((star.xcentroid.values >=3) & (star.xcentroid.values < data.shape[1]-3) & 
#                 (star.ycentroid.values >=3) & (star.ycentroid.values < data.shape[0]-3))
#     star = star.iloc[ind & pos_ind]

#     # -- Generate Aperture photometry for each source -- #
#     x = RoundToInt(star.xcentroid.values); y = RoundToInt(star.ycentroid.values)
#     pos = list(zip(x, y))
#     if len(pos)<1:
#         return None
#     aperture = RectangularAperture(pos, 3.0, 3.0)
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
#     m = sigma_clip(data,masked=True,sigma=5).mask
#     mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
#     aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask)
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
#     aperstats_sky_no_mask = ApertureStats(data, annulus_aperture)
#     aperstats_source = ApertureStats(data, aperture)
#     phot_table = aperture_photometry(data, aperture)
#     phot_table = phot_table.to_pandas()

#     # -- Readout aperture photometry -- #
#     bkg_std = aperstats_sky.std
#     bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
#     bkg_std[bkg_std==0] = np.nanmean(bkg_std[bkg_std>0])
#     negative_ind = aperstats_source.min >= aperstats_sky.median - 3* bkg_std
#     star['sig'] = phot_table['aperture_sum'].values / (aperture.area * bkg_std)
#     star['flux'] = phot_table['aperture_sum'].values
#     star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
#     star['bkgstd'] = aperture.area * bkg_std
#     star = star.iloc[negative_ind]
#     star = star.loc[(star['sig'] > siglim) & (star['bkgstd'] < bkgstd_lim)]

#     if negative:
#         star['flux_sign'] = -1
#         star['flux'] = star['flux'].values * -1
#         star['max_value'] = star['max_value'].values * -1
#     else:
#         star['flux_sign'] = 1

#     return star


# def _Frame_detection(data,prf,frameNum):
#     """
#     Acts on a frame of data. Uses StarFinder to find bright sources, then on each source peform correlation check.
#     """
#     star = None
#     if np.nansum(data) != 0.0:
#         p = _Find_stars(data,prf)
#         n = _Find_stars(deepcopy(data),prf,negative=True)
#         if (p is not None) | (n is not None):
#             star = pd.concat([p,n])
#             star['frame'] = frameNum
#         else:
#             star = None
#     return star



# # ---------- SourceDetect source finding ---------- #
# def _Do_photometry(star,data,siglim=3,bkgstd_lim=50):
#     """
#     Do aperture photometry on each source.
#     """

#     from scipy.signal import fftconvolve
#     from photutils.aperture import RectangularAnnulus, CircularAperture, ApertureStats, aperture_photometry
#     from astropy.stats import sigma_clip, SigmaClip

#     # -- Photometry on source detect stars -- #
#     x = RoundToInt(star.xcentroid.values); y = RoundToInt(star.ycentroid.values)
#     pos = list(zip(x, y))
#     aperture = CircularAperture(pos, 1.5)
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=20,h_out=20)
#     m = sigma_clip(data,masked=True,sigma=5).mask
#     mask = fftconvolve(m, np.ones((3,3)), mode='same') > 0.5
#     aperstats_sky = ApertureStats(data, annulus_aperture,mask = mask,sigma_clip=SigmaClip(sigma=3,cenfunc='median'))
#     annulus_aperture = RectangularAnnulus(pos, w_in=5, w_out=40,h_out=40)
#     aperstats_sky_no_mask = ApertureStats(data, annulus_aperture,sigma_clip=SigmaClip(sigma=3,cenfunc='median'))
#     aperstats_source = ApertureStats(data, aperture)
#     phot_table = aperture_photometry(data, aperture)
#     phot_table = phot_table.to_pandas()

#     # -- Readout results -- #
#     bkg_std = aperstats_sky.std
#     bkg_std[bkg_std==0] = aperstats_sky_no_mask.std[bkg_std==0] # assign a value without mask using a larger area of sky
#     bkg_std[(~np.isfinite(bkg_std)) | (bkg_std == 0)] = 100
#     negative_ind = aperstats_source.min >= aperstats_sky.median - aperture.area * aperstats_sky.std
#     star['sig'] = phot_table['aperture_sum'].values / (aperture.area * aperstats_sky.std)
#     star['flux'] = phot_table['aperture_sum'].values
#     star['mag'] = -2.5*np.log10(phot_table['aperture_sum'].values)
#     star['bkgstd'] = aperture.area * bkg_std #* aperstats_sky.std
    
#     star = star.loc[(star['sig'] >= siglim) & (star['bkgstd'] <= bkgstd_lim)]

#     return star 


# def _Source_detect(flux,cpu,siglim=2,bkgstd=50,maxattempts=5):
#     """
#     Run source detect on each image.
#     """
    
#     from sourcedetect import SourceDetect
#     from joblib import Parallel, delayed 

#     attempt = 0
#     passed = False
#     while (not passed) & (attempt <= maxattempts):
#         try:
#             res = SourceDetect(flux,run=True,train=False).result
#             passed = True
#         except:
#             attempt += 1

#     res = res.drop('psflike',axis=1)

#     frames = res['frame'].unique()
#     stars = Parallel(n_jobs=cpu)(delayed(_Do_photometry)(res.loc[res['frame'] == frame],flux[frame],siglim,bkgstd) for frame in frames)
#     res = pd.concat(stars)

#     ind = (res['xint'].values > 3) & (res['xint'].values < flux.shape[2]-3) & (res['yint'].values >3) & (res['yint'].values < flux.shape[1]-3)
#     res = res[ind]

#     return res




def _Source_mask(res,mask):
    """
    Add source mask value at each source location.
    """

    xInts = res['xint'].values
    yInts = res['yint'].values

    res['source_mask'] = mask[yInts,xInts]

    return res

def _Count_detections(result):
    """
    Count number of source detections per objid.
    """

    ids = result['objid'].values
    unique = np.unique(ids, return_counts=True)
    unique = list(zip(unique[0],unique[1]))

    array = np.zeros_like(ids)

    for id,count in unique:
        index = (result['objid'] == id).values
        array[index] = count

    result['n_detections'] = array

    return result


def _Make_dataframe(results,data):
    """
    Collate results into a dataframe.
    """

    # -- Collates results into a dataframe -- #
    frame = None
    for result in results:
        if frame is None:
            frame = result
        else:
            frame = pd.concat([frame,result])
    
    # -- Again checks to make sure all sources are at least 3px from the edge -- # 
    frame['xint'] = deepcopy(RoundToInt(frame['xcentroid'].values))
    frame['yint'] = deepcopy(RoundToInt(frame['ycentroid'].values))
    ind = ((frame['xint'].values >3) & (frame['xint'].values < data.shape[1]-3) & 
           (frame['yint'].values >3) & (frame['yint'].values < data.shape[0]-3))
    frame = frame[ind]

    return frame

# def _Collate_frame(results):
#     """
#     Collate result dataframe to be more managable.
#     """
    

#     frame_sf = results[results['method'] == 'SF'].copy()
#     frame_sd = results[results['method'] == 'SD'].copy()

#     frame_sf = frame_sf.rename(columns={
#         'xcentroid': 'xcentroid_SF',
#         'ycentroid': 'ycentroid_SF'
#     })
#     frame_sd = frame_sd.rename(columns={
#         'xcentroid': 'xcentroid_SD',
#         'ycentroid': 'ycentroid_SD'
#     })

#     frame_merged = pd.merge(
#         frame_sf,
#         frame_sd,
#         on=['objid', 'frame'],        
#         how='outer',
#         suffixes=('_SFtemp', '_SDtemp'),
#         sort=False
#     )

#     for col in frame_sf.columns:
#         if col in ['objid', 'xcentroid_SF', 'ycentroid_SF']:
#             continue
#         sf_col = f"{col}_SFtemp"
#         sd_col = f"{col}_SDtemp"
#         if sf_col in frame_merged.columns and sd_col in frame_merged.columns:
#             # Prefer SD values when available
#             frame_merged[col] = frame_merged[sd_col].combine_first(frame_merged[sf_col])
#             frame_merged.drop(columns=[sf_col, sd_col], inplace=True)

#     frame_merged = frame_merged.drop(columns=['method_SFtemp', 'method_SDtemp','method'], errors='ignore')

#     frame_merged['xcentroid'] = frame_merged['xcentroid_SD'].combine_first(frame_merged['xcentroid_SF'])
#     frame_merged['ycentroid'] = frame_merged['ycentroid_SD'].combine_first(frame_merged['ycentroid_SF'])

#     frame_merged['xint'] = RoundToInt(frame_merged['xcentroid'])
#     frame_merged['yint'] = RoundToInt(frame_merged['ycentroid'])

#     return frame_merged

def _Brightest_Px(flux,frame):
    """
    Find the brightest pixel.
    """

    brightest_xint = []
    brightest_yint = []
    for _,row in frame.iterrows():
        x = int(row['xint'])
        y = int(row['yint'])
        im = int(row['frame'])
        image = flux[im,y-1:y+2,x-1:x+2]
        iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
        brightesty = y + (iy - 1)  # shift from 3x3 center
        brightestx = x + (ix - 1)
        brightest_xint.append(brightestx)
        brightest_yint.append(brightesty)

    frame['xint_brightest'] = brightest_xint
    frame['yint_brightest'] = brightest_yint

    return frame




# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- # 
# -------------------------------------------------- Event finding functions -------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #


def _Get_temporal_events(df, max_gap=2, frame_col='frame', id_col='eventid',startingID=1):
    """
    Labels temporally clustered events in a DataFrame by assigning an event ID.

    Parameters:
        df: Pandas DataFrame with a time/frame column
        max_gap: maximum allowed gap (in frames) between temporally connected sources
        frame_col: name of the column containing frame/time info
        id_col: name of the new event ID column to add

    Returns:
        Modified DataFrame with a new column for event ID
    """
    df = df.copy()
    df = df.sort_values(frame_col).reset_index(drop=False)  # preserve original index

    event_ids = []
    current_event = startingID
    event_ids.append(current_event)

    for i in range(1, len(df)):
        frame_diff = df.loc[i, frame_col] - df.loc[i - 1, frame_col]
        if frame_diff <= max_gap:
            event_ids.append(current_event)
        else:
            current_event += 1
            event_ids.append(current_event)

    df[id_col] = event_ids
    df = df.sort_values('index').set_index('index')  # return to original order
    return df


def _Lightcurve_significance(time,flux,frame_start,frame_end,pos,flux_sign,
                           event_mask=None,event_time_buffer = 0.2,calc_time_window=1):

    from tessellate.tools import Generate_LC
    from scipy import stats

    y = pos[1] 
    x = pos[0] 
    t,lc = Generate_LC(time,flux,x,y,radius=1.5)

    time_start = time[frame_start]
    time_end = time[frame_end]
    
    buffer_start = np.argmin(abs(time - (time_start - event_time_buffer)))
    buffer_end = np.argmin(abs(time - (time_end + event_time_buffer)))
    if buffer_start < 0:
        buffer_start = 0
    if buffer_end > len(time):
        buffer_end = len(time) - 1 

    window_start = np.argmin(abs(time - (time_start - event_time_buffer-calc_time_window)))
    window_end = np.argmin(abs(time - (time_end + event_time_buffer+calc_time_window)))
    if window_start < 0:
        window_start = 0
    if window_end > len(time):
        window_end = len(time) - 1 

    frames = np.arange(0,len(lc))
    if event_mask is None:
        event_mask = np.ones_like(frames).astype(bool)

    ind = event_mask & (((frames > window_start) & (frames < buffer_start)) | ((frames < window_end) & (frames > buffer_end)))
    #mean,med, std = sigma_clipped_stats(lc[ind])
    
    t_window = t[ind]
    lc_window = lc[ind]
    
    med = np.nanmedian(lc_window)
    std = np.nanstd(lc_window)

    lcevent = lc[frame_start:frame_end+1]
    lc_sig = (lcevent - med) / std

    if flux_sign >= 0:
        sig_max = np.nanmax(lc_sig)
        sig_med = np.nanmean(lc_sig)
        max_flux = np.nanmax(lcevent)
        max_frame = np.argmax(lcevent)+frame_start
        
    else:
        max_flux = np.nanmin(lcevent)
        max_frame = np.argmin(lcevent)+frame_start
        sig_max = abs(np.nanmin(lc_sig))
        sig_med = abs(np.nanmean(lc_sig))
    
    lc_sig = (lc - med) / std

    if ind.sum() < 2:
        return np.nan, np.nan, np.ones_like(time)*np.nan,max_flux,max_frame,False

    slope, _, _, _, _ = stats.linregress(t_window, lc_window)
    baseline_is_flat =(abs(slope) / std < 2) and  (abs(med) < std) and (std < 5)

    return sig_max, sig_med, lc_sig * flux_sign, max_flux, max_frame, baseline_is_flat


def _Lightcurve_event_checker(lc_sig,triggers,siglim=3,maxsep=5):

    triggers = list(np.unique(triggers))
    start = np.nanmin(triggers)
    end = np.nanmax(triggers)

    sig_ind = np.where(lc_sig>= siglim)[0]

    new_frames = (frame for frame in sig_ind if start <= frame <= end and frame not in triggers)
    triggers.extend(new_frames)

    doneBack = False
    frame_ind = start-1
    count = 0
    while not doneBack:
        if frame_ind in sig_ind:
            triggers.append(frame_ind)
        else:
            count += 1
            if count == maxsep:
                doneBack = True
        frame_ind -= 1


    doneFront = False
    frame_ind = end+1
    count = 0
    while not doneFront:
        if frame_ind in sig_ind:
            triggers.append(frame_ind)
        else:
            count += 1
            if count == maxsep:
                doneFront = True
        frame_ind += 1

    new_start = np.nanmin(triggers)
    new_end = np.nanmax(triggers)
    n_detections = len(triggers)

    return new_start,new_end,n_detections,sorted(triggers)


def _Fit_psf(flux, event, prf, frames, uncertainty_funcs, exposure_time, big_size=15, small_size=5,core_size=3):
    """
    Generate an cutout around an event and fit PSF. 
    Chooses the frame based on the highest SNR between stack through event and individual frames.
    """

    from .localisation import PSF_Fitter
    from astropy.stats import sigma_clipped_stats

    x0 = RoundToInt(event['xcentroid_det'])
    y0 = RoundToInt(event['ycentroid_det'])
    sign = event['flux_sign']

    half_big = big_size // 2
    half_small = small_size // 2
    half_core = core_size // 2

    h, w = flux.shape[1], flux.shape[2]

    frames = np.unique(frames)

    # -- Generate a stacked 3x3 aperture to identify and lock onto brightest pixel -- #
    stacked_flux_3x3 = np.zeros((3, 3), dtype=np.float32) 
    for i in frames: 
        y1, y2 = y0 - 1, y0 + 2 
        x1, x2 = x0 - 1, x0 + 2 
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w: 
            continue 
        cut = flux[i, y1:y2, x1:x2] * sign 
        stacked_flux_3x3 += cut 
        
    iy, ix = np.unravel_index(np.nanargmax(stacked_flux_3x3), stacked_flux_3x3.shape)

    brightest_y = y0 + (iy - 1)
    brightest_x = x0 + (ix - 1)

    event['xint_brightest'] = brightest_x
    event['yint_brightest'] = brightest_y


    # --- Produce big_size cutouts with NaN padding and add to stack --- #
    stacked_big = np.zeros((big_size, big_size), dtype=np.float32)
    cuts = []
    snrs = []

    y1 = brightest_y - half_big        # Desired bounds in full image
    y2 = brightest_y + half_big + 1
    x1 = brightest_x - half_big
    x2 = brightest_x + half_big + 1

    yy1, yy2 = max(0, y1), min(h, y2)   # Clip to image bounds
    xx1, xx2 = max(0, x1), min(w, x2)

    for i in frames:
        cut = np.full((big_size, big_size), np.nan, dtype=np.float32)   # Create NaN-padded cut

        cy1 = yy1 - y1
        cy2 = cy1 + (yy2 - yy1)
        cx1 = xx1 - x1
        cx2 = cx1 + (xx2 - xx1)

        cut[cy1:cy2, cx1:cx2] = flux[i, yy1:yy2, xx1:xx2] * sign

        valid = cut[~np.isnan(cut)]     # Compute noise only on valid pixels
        if valid.size == 0:
            continue
        _, _, noise = sigma_clipped_stats(valid, sigma=3)

        core = cut[                         
            half_big-half_core:half_big+half_core+1,      # core_size x core_size aperture around core
            half_big-half_core:half_big+half_core+1,
        ]

        flux_sum = np.nansum(core)      # Estimate the snr from core
        npix = core_size ** 2            # NOTE: core_size here should be the side length (3), not sqrt(npix)
        background_term = npix * noise**2
        poisson_term = flux_sum / exposure_time   # flux_sum in e-/s, exptime_s the effective exposure time
        ap_err = np.sqrt(background_term + poisson_term)
        snr = flux_sum / ap_err

        cuts.append(cut)
        snrs.append(snr)
        stacked_big += np.nan_to_num(cut, nan=0.0)

    if len(cuts) == 0:
        raise ValueError("All PSF cutouts were out of bounds.")


    # --- Stacked SNR (ignore NaNs) --- #
    valid = stacked_big[~np.isnan(stacked_big)]
    _, _, noise = sigma_clipped_stats(valid, sigma=3)

    stacked_small = stacked_big[
        half_big-half_small:half_big+half_small+1,
        half_big-half_small:half_big+half_small+1,
    ]

    stacked_core = stacked_big[
        half_big-half_core:half_big+half_core+1,
        half_big-half_core:half_big+half_core+1,
    ]

    stacked_flux_sum = np.nansum(stacked_core)      # Estimate the snr from core
    background_term = npix * noise**2
    poisson_term = stacked_flux_sum / exposure_time   # flux_sum in e-/s, exptime_s the effective exposure time
    stacked_ap_err = np.sqrt(background_term + poisson_term)
    stacked_snr = stacked_flux_sum / stacked_ap_err

    # --- Choose best image or stacked through event --- #
    if np.max(snrs) >= stacked_snr:
        idx = int(np.argmax(snrs))
        centred_flux = cuts[idx][
            half_big-half_small:half_big+half_small+1,
            half_big-half_small:half_big+half_small+1,
        ]
        snr = snrs[idx]
        stacked_psf_fit = 0
    else:
        centred_flux = stacked_small
        snr = stacked_snr
        stacked_psf_fit = 1 

    # --- PSF fit --- #
    unc_x = uncertainty_funcs[0](snr)
    unc_y = uncertainty_funcs[1](snr)

    fitter = PSF_Fitter(small_size, prf)
    fitter.fit_psf(centred_flux, limx=0.5, limy=0.5)

    event['xcentroid_psf'] = fitter.source_x + brightest_x
    event['ycentroid_psf'] = fitter.source_y + brightest_y
    event['xcentroid_err_psf'] = unc_x
    event['ycentroid_err_psf'] = unc_y
    event['snr_psf'] = snr

    r = np.corrcoef(centred_flux.flatten(), fitter.psf.flatten())[0, 1]
    event['psf_like'] = r

    norm_flux = centred_flux / np.nansum(centred_flux)
    event['psf_diff'] = np.nansum(np.abs(norm_flux - fitter.psf))
    event['psf_stacked'] = stacked_psf_fit

    return event



def _Isolate_events(objid,time,flux,sources,sector,cam,ccd,cut,prf,
                    exposure_time,snr_to_localisation_func,nan_frames,
                    frame_buffer,event_time_buffer,calc_time_window):
    """
    Groups sources for given objid into temporally separated events.
    """

    from .tools import pandas_weighted_avg

    # -- Select all sources grouped to this objid -- #
    source = sources[sources['objid']==objid]
    
    all_labelled_sources = []

    # -- For this objid, separate positive and negative detections -- #
    startingID=1
    for flux_sign in source['flux_sign'].unique():
        signed_sources = source[source['flux_sign']==flux_sign]

        # weighted_signedsources = pandas_weighted_avg(signed_sources)

        # # -- Look for any regular variability -- #
        # peak_freq, peak_power = _Fit_period(time,flux,weighted_signedsources.iloc[0])

        # # -- Run RFC Classification -- #
        # cf_classification, cf_prob = _Check_classifind(time,flux,weighted_signedsources.iloc[0])

        labelled_sources = _Get_temporal_events(signed_sources,max_gap=frame_buffer,startingID=startingID)
        all_labelled_sources.append(labelled_sources)
        startingID = np.nanmax(labelled_sources['eventid'])+1
    
    all_labelled_sources = pd.concat(all_labelled_sources, ignore_index=True)
    
    # -- Iterate through eventids -- #
    dfs = []
    for eventID in all_labelled_sources['eventid'].unique():
        event = {}
        eventsources = deepcopy(all_labelled_sources[all_labelled_sources['eventid']==eventID])
        weighted_eventsources = pandas_weighted_avg(eventsources,'snr')

        flux_sign = int(eventsources.iloc[0]['flux_sign'])

        frame_start = int(eventsources['frame'].min())
        frame_end = int(eventsources['frame'].max())
        xint = RoundToInt(weighted_eventsources.iloc[0]['xint_brightest'])
        yint = RoundToInt(weighted_eventsources.iloc[0]['yint_brightest'])

        # -- Calculate significance of detection above background and local light curve -- #
        _, _, sig_lc, _, _, _ = _Lightcurve_significance(time,flux,frame_start,
                                                      frame_end,[xint,yint],flux_sign,
                                                        event_time_buffer=event_time_buffer,
                                                        calc_time_window=calc_time_window)
        

        sig_lc[nan_frames] = np.nan
        
        # -- Extend the event duration based on time spent above significance threshold -- #
        frame_start,frame_end,n_detections,frames = _Lightcurve_event_checker(sig_lc,eventsources['frame'].values,siglim=3,maxsep=5)

        # -- Initialise event information -- #
        event['frame_bin'] = int(eventsources.iloc[0].frame_bin)
        event['objid'] = int(objid)
        event['eventid'] = int(eventID)
        event['sector'] = int(sector)
        event['camera'] = int(cam)
        event['ccd'] = int(ccd)
        event['cut'] = int(cut)
        event['frame_start'] = int(frame_start)
        event['frame_end'] = int(frame_end)
        event['frame_duration'] = int(event['frame_end']-event['frame_start']+1)
        event['flux_sign'] = flux_sign
        event['n_detections'] = int(n_detections)
        # event['bkg_level'] = weighted_eventsources.iloc[0]['bkg_level']
        # event['bkg_std'] = weighted_eventsources.iloc[0]['bkg_std']
        event['neg_extent'] = weighted_eventsources.iloc[0]['neg_extent']
        event['fwhm'] = weighted_eventsources.iloc[0]['fwhm']
        event['ellipticity'] = weighted_eventsources.iloc[0]['ellipticity']
        event['image_sig_max'] = np.nanmax(eventsources['snr'].values)
        event['xcentroid_det'] = weighted_eventsources.iloc[0]['xcentroid']
        event['ycentroid_det'] = weighted_eventsources.iloc[0]['ycentroid']

        # -- Fit PSF -- #
        event = _Fit_psf(flux,event,prf,frames,snr_to_localisation_func,exposure_time)
        
        # -- If event is quite PSF-like, centroid likely good -- #
        if event['psf_like']>0.5:
            event['xcentroid'] = event['xcentroid_psf']
            event['ycentroid'] = event['ycentroid_psf']
            event['xcentroid_err'] = event['xcentroid_err_psf']
            event['ycentroid_err'] = event['ycentroid_err_psf']
        else:
            event['xcentroid'] = event['xcentroid_det']
            event['ycentroid'] = event['ycentroid_det']
            event['xcentroid_err'] = 0.5
            event['ycentroid_err'] = 0.5

        event['xint'] = RoundToInt(event['xcentroid'])
        event['yint'] = RoundToInt(event['ycentroid'])

        # # -- Calculate and save LC statistics -- #
        # sig_max, sig_med, _, max_flux, max_frame = _Check_LC_significance(time,flux,
        #                                                         event['frame_start'],event['frame_end'],
        #                                                         [event['xint'],event['yint']],
        #                                                         sign,buffer=buffer,base_range=base_range)

        # event['frame_max'] = int(max_frame)
        # event['flux_max'] = int(max_flux)
        # event['lc_sig_max'] = sig_max
        # event['lc_sig_med'] = sig_med

        # -- Miscellaneous info -- #
        # event['peak_freq'] = peak_freq[0]
        # event['peak_power'] = peak_power[0]
        event['source_mask'] = eventsources.iloc[0]['source_mask']
        # event['cf_class'] = cf_classification
        # event['cf_prob'] = cf_prob

        df = pd.DataFrame([event])
        dfs.append(df)

    events_df = pd.concat(dfs, ignore_index=True)

    # -- Add to each dataframe row the number of events in the total object -- #
    events_df['total_events'] = int(len(events_df))
    
    return events_df 

def _Extract_Lightcurve_Properties(time,flux,events,event_time_buffer,calc_time_window):

    events = deepcopy(events)
    events = events.reset_index(drop=True) 

    events['classification'] = '-'
    for objid in np.unique(events.objid):
        evs = events[events.objid==objid]

        event_mask = np.zeros_like(time)
        for i in range(len(evs)):
            ev = evs.iloc[i]
            event_mask[int(ev.frame_start):int(ev.frame_end)+1] = i+1
        
        i = 0
        for idx,ev in evs.iterrows():
            i += 1

            mask = (event_mask == i) | (event_mask == 0)

            frame_start = int(ev.frame_start)
            frame_end = int(ev.frame_end)
            xint = int(ev.xint)
            yint = int(ev.yint)

            sig_max, sig_med, sig_lc, max_flux, max_frame, flat_baseline = _Lightcurve_significance(time,flux,
                                                                            frame_start,frame_end,
                                                                            [xint,yint],ev.flux_sign,
                                                                            event_mask = mask,
                                                                            event_time_buffer=event_time_buffer,
                                                                            calc_time_window=calc_time_window)
            
            events.loc[idx, 'frame_max'] = int(max_frame)
            events.loc[idx, ['flux_max', 'lc_sig_max', 'lc_sig_med','lc_flat']] = (max_flux, sig_max, sig_med,flat_baseline)

            # -- Tag cosmic rays -- #
            if (ev.frame_bin == 1) & (ev.flux_sign == 1):
                event_sig_lc = sig_lc[frame_start:frame_end+1]

                # If only one frame in event, if bright = cosmic ray, if faint = junk
                if ev.frame_duration == 1:
                    if len(event_sig_lc[event_sig_lc >= 5]) == 1: 
                        events.loc[idx,'classification'] = 'CosmicRay'
                    else:
                        events.loc[idx,'classification'] = 'Junk'
                
                # If two image plane detections:
                # but only 1 frame above 3sigma / 2 frames above 3sigma but not consecutive
                elif ev.n_detections == 2:

                    if (len(event_sig_lc[event_sig_lc >= 5]) == 1) & (len(event_sig_lc[event_sig_lc >= 3]) == 1):
                        events.loc[idx,'classification'] = 'CosmicRay'

                    elif ev.frame_duration > 2:
                        # if (len(event_sig_lc[event_sig_lc >= 5]) == 2) & (len(event_sig_lc[event_sig_lc >= 3]) == 2):
                        sorted_idx = np.argsort(event_sig_lc)[::-1]
                        if abs(np.diff(sorted_idx)[0]) > 1:

                            if (len(event_sig_lc[event_sig_lc >= 5]) > 0):
                                events.loc[idx,'classification'] = 'CosmicRay'
                            else:
                                events.loc[idx,'classification'] = 'Junk'
                    
    return events 
            

def _Flag_Event_Frames(events,max_frame,std=7):

    from astropy.stats import sigma_clipped_stats

    events = deepcopy(events)

    max_frames = (events.frame_max * events.frame_bin).values
    max_frames = np.clip(max_frames,a_min=0,a_max=max_frame)
    frames,vals = np.unique(max_frames,return_counts=True)

    # median_count = np.nanmedian(vals)
    # std_count = np.nanstd(vals)

    _,median_count,std_count = sigma_clipped_stats(vals,sigma=7)

    extreme_frames = frames[np.where(vals>median_count+std*std_count)]

    
    bad_idx = np.where(np.isin(max_frames, extreme_frames))[0].astype(int)

    events['bad_frame_flag'] = 0
    events.loc[events.index[bad_idx],'bad_frame_flag'] = 1

    return events

def _Recheck_asteroid_lcs(time,flux,events):
    """
    Asteroids move their centroids a lot in PSF fitting, so often their light curve is now off. This correts for that.
    """

    asteroids = events[events['classification']=='Asteroid']
    if len (asteroids) > 0:
        for _, ast in asteroids.iterrows():
            objid = int(ast['objid'])
            eventid = int(ast['eventid'])
            frame_bin = int(ast['frame_bin'])
            xint = int(ast['xint'])
            yint = int(ast['yint'])
            frame_start = int(ast['frame_start'])
            frame_end = int(ast['frame_end'])

            evs = events[events.objid==objid]

            event_mask = np.zeros_like(time)
            for _, ev in evs.iterrows():
                event_mask[int(ev.frame_start):int(ev.frame_end)+1] = ev.eventid
            
            mask = (event_mask == eventid) | (event_mask == 0)
            
            _, _, lc_sig, _, _, _ = _Lightcurve_significance(
                time, flux, frame_start, frame_end, [xint, yint], flux_sign=1, 
                event_mask=mask, event_time_buffer=0.2, calc_time_window=0.5
            )
            

            start, end, _, _ = _Lightcurve_event_checker(
                lc_sig, np.arange(frame_start, frame_end+1)
            )

            n_detections = np.nansum([frame_start - start, end - frame_end])

            # Update self.events
            event_mask = (events['objid'] == objid) & (events['eventid'] == eventid) &  (events['frame_bin'] == frame_bin)
            events.loc[event_mask, 'frame_start'] = start
            events.loc[event_mask, 'frame_end'] = end
            events.loc[event_mask, 'mjd_start'] = time[start]
            events.loc[event_mask, 'mjd_end'] = time[end]
            events.loc[event_mask, 'frame_duration'] = end - start
            events.loc[event_mask, 'mjd_duration'] = time[end] - time[start]
            events.loc[event_mask, 'n_detections'] += n_detections

    return events


def _Crossbin_group_worker(asteroid_group_id, group_events):

    """
    Process a single asteroid_crossbin_group chunk independently.
    Returns a one-column DataFrame (crossbin_ids) with the same index.
    """

    group_events = group_events.copy()
    group_events['crossbin_ids'] = [[] for _ in range(len(group_events))]

    for cg in np.unique(group_events['crossbin_group']):
        group_df = group_events[group_events['crossbin_group'] == cg].sort_values('frame_bin')

        for i, row in group_df.iterrows():
            if not group_events.at[i, 'crossbin_ids']:
                group_events.at[i, 'crossbin_ids'] = [i]

            current_ids = group_events.at[i, 'crossbin_ids']

            if row['classification'] == 'Asteroid':
                coarser = group_events[
                    (group_events['asteroid_crossbin_group'] == asteroid_group_id) &
                    (group_events['frame_bin'] > row['frame_bin'])
                ].sort_values('frame_bin')
            else:
                coarser = group_df[group_df['frame_bin'] > row['frame_bin']]

            mask = (
                (row['frame_start'] * row['frame_bin'] <= coarser['frame_end'] * coarser['frame_bin'] + coarser['frame_bin']) &
                (row['frame_end'] * row['frame_bin'] >= coarser['frame_start'] * coarser['frame_bin'] - coarser['frame_bin'])
            )
            for j in coarser[mask].index:
                merged = list(set(group_events.at[j, 'crossbin_ids'] + current_ids))
                group_events.at[j, 'crossbin_ids'] = merged

    return group_events[['crossbin_ids']]


class Detector():

    def __init__(self,sector,cam,ccd,data_path='/fred/oz335/TESSdata',n=8,
                 match_variables=True,mode='both',part=None,injection=False,injection_dir='source_injection',
                 cpu=multiprocessing.cpu_count()):
        
        """
        Tessellate Detection Class.
        """

        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.data_path = data_path
        self.n = n
        self.match_variables = match_variables
        self.mode = mode

        self.cpu = cpu

        self.flux = None
        self.ref = None
        self.time = None
        self.mask = None
        self.sources = None  #raw detection results
        self.events = None   #temporally located with same object id
        self.objects = None   #temporally and spatially combined
        self.cut = None
        self.bkg = None

        self._inj_path = injection_dir if injection else '.'

        if part is None:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}'
        elif part == 1:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part1'
        elif part == 2:
            self.path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Part2'
        else:
            e = 'Invalid Part Parameter!'
            raise AttributeError(e)


        
    # ------------------------------ Gathering data and cached results ------------------------------ #

    def gather_results(self,cut):
        """
        Gather pre-existing sources / events csvs.
        """

        import os

        self.objects = None
        path = f'{self.path}/Cut{cut}of{self.n**2}/{self._inj_path}'

        if table_exists(f'{path}/detected_sources.csv'):
            self.sources = load_table(f'{path}/detected_sources.csv')    # raw detection results
        else:
            print('No detected sources file found',flush=True)
            self.sources = None

        if table_exists(f'{path}/detected_events.csv'):
            self.events = load_table(f'{path}/detected_events.csv')    # raw detection results
        else:
            print('No detected events file found',flush=True)
            self.events = None
            

    def gather_data(self,cut,flux=True,time=True,bkg=False,mask=False,ref=False,verbose=True):
        """
        Gather reduced data.
        """

        from .localisation import CutWCS

        if verbose:
            ts = clock()
            print(f'Loading Cut {cut} Data...',end='\r',flush=True)

        base = f'{self.path}/Cut{cut}of{self.n**2}/{self._inj_path}/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{cut}_of{self.n**2}'

        if flux:
            self.flux = np.load(base + '_ReducedFlux.npy')
            self.cut = cut

        if time:
            self.time = np.load(base + '_Times.npy')

        if ref:
            self.ref = np.load(base + '_Ref.npy')

        if bkg:
            self.bkg = np.load(base + '_Background.npy')

        if mask:
            self.mask = np.load(base + '_Mask.npy')

        # # Load orbit ref data if available
        # import os
        # seg_path = base + '_OrbitSegments.npy'
        # ref_path = base + '_OrbitRefs.npz'
        # if os.path.exists(seg_path) and os.path.exists(ref_path):
        #     self.orbit_segments = np.load(seg_path)
        #     raw = np.load(ref_path)
        #     self.orbit_refs = {int(k): raw[k] for k in raw.files}
        # else:
        #     self.orbit_segments = None
        #     self.orbit_refs = None

        self.wcs = CutWCS(self.data_path,self.sector,self.cam,self.ccd,cut=cut,n=self.n)

        if verbose:
            print(f'Loading Cut {cut} Data -- done! ({clock()-ts:.0f}s)',flush=True)

    


    # ------------------------------ Source finding functions ------------------------------ #

    def _find_sources_in_images(self,flux,#column,row,
                                inputNums=None,isolate_single_detections=True,min_snr=3.0):#,datadir='/fred/oz335/_local_TESS_PRFs'):
        """
        Detect sources in flux and collate information.
        """

        from joblib import Parallel, delayed 
        from tqdm import tqdm
        # from PRF import TESS_PRF

        if inputNums is not None:
            flux = flux[inputNums]
            inputNum = inputNums[-1]
        else:
            inputNum = 0
                
        # if self.sector < 4:
        #     prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{datadir}/Sectors1_2_3')
        # else:
        #     prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=f'{datadir}/Sectors4+')

        # -- Run source detection on each frame -- #
        length = np.linspace(0,flux.shape[0]-1,flux.shape[0]).astype(int)
        # if self.mode == 'starfind':
        #     sources = Parallel(n_jobs=self.cpu)(delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
            
        #     sources = _Make_dataframe(sources,flux[0])
        #     sources['method'] = 'starfind'

        # elif self.mode == 'sourcedetect':
        #     sources = _Source_detect(flux,self.cpu)
        #     sources['method'] = 'sourcedetect'
        #     sources = sources[~pd.isna(sources['xcentroid'])]

        # elif self.mode == 'both':
        #     results = Parallel(n_jobs=self.cpu)( delayed(_Frame_detection)(flux[i],prf,inputNum+i) for i in tqdm(length))
            
        #     star = _Make_dataframe(results,flux[0])
        #     star['method'] = 'starfind'

        #     machine = _Source_detect(flux,self.cpu)
        #     machine = machine[~pd.isna(machine['xcentroid'])]
        #     machine['method'] = 'sourcedetect'

        #     sources = pd.concat([star.assign(method='SF'), machine.assign(method='SD')], ignore_index=True)

        # elif self.mode == 'claudefinder':
        # print('Source Finding')
        sources = Parallel(n_jobs=self.cpu)(delayed(_TESS_sourcefinder)(flux[i],inputNum+i,min_snr) for i in tqdm(length,desc='Source Finding'))

        # print('Making DataFrame')
        sources = _Make_dataframe(sources,flux[0])

        # print('Grouping Spatially')
        # -- Group based on distance -- #
        sources = _Spatial_group(sources,distance=0.5,min_samples=2)       

        # # -- Prefer SourceDetect results -- #
        # sources = _Collate_frame(sources)                  

        # sources = sources.drop('objid',axis=1)

        # -- Group based on distance -- #
        # sources = _Spatial_group(sources,distance=0.5,min_samples=2)   # prefer SourceDetect results

        # -- Separate source detections with no companion -- #
        single_isolated_detections = None
        if isolate_single_detections:
            single_isolated_detections = sources[sources['objid']==0]
            sources = sources[sources['objid']>0].reset_index(drop=True)

        # print('Brightest Px')
        # --  Find brightest pixels around each source -- #
        sources = _Brightest_Px(flux,sources)               

        # print('Brightest Px')
        # -- Add in TessReduce source mask value -- #
        sources = _Source_mask(sources,self.mask)                

        # -- Count num detections for each objid -- #
        sources = _Count_detections(sources)               

        return sources,single_isolated_detections


    def _wcs_time_info(self,result):
        """
        Physical units for sources.
        """
        
        result['ra'],result['dec'] = self.wcs.all_pix2world(result['xcentroid'],result['ycentroid'],0)
        trueframes = result['frame']*result['frame_bin']
        trueframes[trueframes>=len(self.time)] = len(self.time)-1
        result['mjd'] = self.time[trueframes]
    
        return result

    def _run_find_sources(self,frame_bin,min_snr):
        """
        Run the source finding on this specific frame_binning.
        """

        from .dataprocessor import DataProcessor
        # from .catalog_queries import match_result_to_cat #,find_variables, gaia_stars,
        from .tools import pandas_weighted_avg,Frame_Bin
        
        save_folder = f'{self.path}/Cut{self.cut}of{self.n**2}/{self._inj_path}'

        # -- Access information about the cut with respect to the original ccd -- #        
        processor = DataProcessor(sector=self.sector,data_path=self.data_path,verbose=2)
        cut_corners, cut_centre_px, _, _ = processor.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        # column = cut_centre_px[self.cut-1][0]
        # row = cut_centre_px[self.cut-1][1]

        # -- Rebin time and flux by given factor -- #
        if frame_bin > 1:
            time,flux = Frame_Bin(self.sector,self.cam,self.time,self.flux,frame_bin)
            isolate_single_detections = False
        else:
            time = self.time; flux = self.flux
            isolate_single_detections = True

        # -- Run the detection algorithm, generates dataframe -- #
        results,single_isolated_detections = self._find_sources_in_images(flux,#column=column,row=row,datadir=self.prf_path,
                                                                 isolate_single_detections=isolate_single_detections,min_snr=min_snr)
        
        # -- Save out single detections which are isolated in space in time, probably noise, maybe cool -- #
        if isolate_single_detections:
            single_isolated_detections.to_csv(f'{save_folder}/single_isolated_detections.csv',index=False)
        
        results['frame_bin'] = frame_bin

        # -- Add wcs, time, ccd info to the results dataframe -- #
        results = self._wcs_time_info(results)
        results['xccd'] = deepcopy(results['xcentroid'] + cut_corners[self.cut-1][0])
        results['yccd'] = deepcopy(results['ycentroid'] + cut_corners[self.cut-1][1])
        
        # -- For each source, finds the average position based on the weighted average of the flux -- #
        # av_var = pandas_weighted_avg(results[['objid','sig','xcentroid','ycentroid','ra','dec','xccd','yccd']])
        # av_var = av_var.rename(columns={'xcentroid':'x_source',
        #                                 'ycentroid':'y_source',
        #                                 'e_xcentroid':'e_x_source',
        #                                 'e_ycentroid':'e_y_source',
        #                                 'ra':'ra_source',
        #                                 'dec':'dec_source',
        #                                 'e_ra':'e_ra_source',
        #                                 'e_dec':'e_dec_source',
        #                                 'xccd':'xccd_source',
        #                                 'yccd':'yccd_source',
        #                                 'e_xccd':'e_xccd_source',
        #                                 'e_yccd':'e_yccd_source'})
        # av_var = av_var.drop(['sig','e_sig'],axis=1)
        # results = results.merge(av_var, on='objid', how='left')

        # -- Calculates the background stddev for each source -- #        
        f = results['frame'].values
        x = results['xint'].values
        y = results['yint'].values

        r_outer = 11
        r_inner = 2

        dy, dx = np.mgrid[-r_outer:r_outer+1, -r_outer:r_outer+1]
        mask = (dy**2 + dx**2) > r_inner**2
        dy = dy[mask].ravel()
        dx = dx[mask].ravel()

        H, W = flux.shape[1], flux.shape[2]

        yy = (y[np.newaxis, :] + dy[:, np.newaxis]).clip(0, H - 1)
        xx = (x[np.newaxis, :] + dx[:, np.newaxis]).clip(0, W - 1)

        patch = flux[f[np.newaxis, :], yy, xx]

        results['bkg_std'] = np.nanstd(patch, axis=0)

        # results['bkg_level'] = 0
        # if self.bkg is not None:
        #     f = results['frame'].values
        #     x = results['xint'].values; y = results['yint'].values
        #     b = []
        #     for i in range(3):
        #         i-=1
        #         for j in range(3):
        #             j-=1
        #             b += [self.bkg[f,y-i,x+j]]
        #     b = np.array(b)
        #     b = np.nansum(b,axis=0)
        #     results['bkg_level'] = b

        return results
    
    def _parse_time_bins(self, bins):
        import re

        conversions = {'sec':1/86400,'min': 1/1440, 'hr': 1/24, 'day': 1}
        max_resolution = np.nanmedian(np.diff(self.time))
        
        frame_bins = []
        for bin in bins:
            v, u = re.match(r'([\d.]+)(\w+)', bin).groups()
            resolution = float(v) * conversions[u]
            frame_bins.append(RoundToInt(resolution/max_resolution))

        return frame_bins

    def find_sources(self,time_bins,min_snr):
        """
        Find sources.
        """

        # -- Turn time_bins into frame_bins -- #
        frame_bins = self._parse_time_bins(time_bins)

        # -- Iteratate over frame_bins and run source detection -- #
        sources = pd.DataFrame()
        for frame_bin in frame_bins:
            ts = clock()
            sources = pd.concat([sources,self._run_find_sources(frame_bin,min_snr=min_snr)]) 
            print(f'    Run with time bin = {frame_bin} ({clock()-ts:.0f}s)',flush=True)
            
        # -- Reset objid so each objid,frame_bin pair is unique -- #
        matched = sources[sources['objid'] != 0]
        unmatched = sources[sources['objid'] == 0]
        matched['objid'] = pd.factorize(matched['objid'].astype(str) + '_' + matched['frame_bin'].astype(str))[0]+1
        unmatched['objid'] = np.arange(matched['objid'].max() + 1, matched['objid'].max() + 1 + len(unmatched))

        sources = pd.concat([matched, unmatched])

        self.sources = sources

        self.sources.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/{self._inj_path}/detected_sources.csv',index=False)




    # ------------------------------ Event finding functions ------------------------------ #

    def _get_all_independent_events(self,frame_buffer=5,event_time_buffer=0.2,calc_time_window=0.5):
        """
        Isolate sources into individual temporal events.
        """

        from joblib import Parallel, delayed 
        from tqdm import tqdm
        from .dataprocessor import DataProcessor
        from .localisation import get_snr_to_localisation_func, get_wcs_uncertainty
        from .tools import Frame_Bin
        from PRF import TESS_PRF
        from astropy.io import fits
        
    
        # -- Generate PRF -- #
        dp = DataProcessor(self.sector,data_path=self.data_path)
        cutCornerPx, cutCentrePx, _, _ = dp.find_cuts(cam=self.cam,ccd=self.ccd,n=self.n,plot=False)
        column = cutCentrePx[self.cut-1][0]
        row = cutCentrePx[self.cut-1][1]
        prf = TESS_PRF(self.cam,self.ccd,self.sector,column,row,localdatadir=self.prf_path)
        
        # -- Retrieve cut localisation quality -- #
        snr_to_localisation = get_snr_to_localisation_func(self.prf_path,self.sector,self.cam,self.ccd,self.cut,self.n)
        exposure_time = fits.open(f'{self.path}/wcs/ref/corrected.fits')[1].header['EXPOSURE'] * 86400  # in seconds

        # -- Iterate over frame bins -- #
        frame_bins = np.unique(self.sources.frame_bin)
        events = pd.DataFrame()
        for frame_bin in frame_bins:
            sources = self.sources[self.sources.frame_bin == frame_bin]
            time,flux = Frame_Bin(self.sector,self.cam,self.time,self.flux,frame_bin)

            has_data = np.any(np.isfinite(flux), axis=(1, 2))      # Identify nan_frames 
            nan_frames = np.where(~has_data)[0]

            # -- Iterate over objids to separate into discrete events -- #
            objids = np.unique(sources['objid'].values).astype(int)
            if self.cpu > 1:
                length = np.arange(0,len(objids)).astype(int)
                bin_events = Parallel(n_jobs=self.cpu)(delayed(_Isolate_events)(objids[i],time,flux,sources,
                                                                    self.sector,self.cam,self.ccd,self.cut,prf,exposure_time,snr_to_localisation,nan_frames,
                                                                    frame_buffer,event_time_buffer,calc_time_window) for i in tqdm(length,desc=f'Isolating Bin {frame_bin} Events'))
            else:            
                bin_events = []
                for objid in objids:
                    e = _Isolate_events(objid,time,flux,sources,self.sector,self.cam,
                                        self.ccd,self.cut,prf,exposure_time,snr_to_localisation,nan_frames,
                                        frame_buffer=frame_buffer,buffer=event_time_buffer,calc_time_window=calc_time_window)
                    bin_events += [e]
            
            # -- Extract light curve properties -- #
            bin_events = _Extract_Lightcurve_Properties(time,flux,pd.concat(bin_events,ignore_index=True),event_time_buffer,calc_time_window)

            events = pd.concat([events,bin_events],ignore_index=True)

        events['frame_max'] = events['frame_max'].astype(int)
        events['lc_flat'] = events['lc_flat'].astype(int)

        # -- Provide CCD-relative location -- #
        events['xccd'] = RoundToInt(events['xint'] + cutCornerPx[self.cut-1][0])
        events['yccd'] = RoundToInt(events['yint'] + cutCornerPx[self.cut-1][1])

        # -- Pull the uncertainty on WCS and combine with PSF fit centroid error -- #
        wcs_unc = get_wcs_uncertainty(self.data_path,self.sector,self.cam,self.ccd,self.cut,self.n)
        if np.isnan(wcs_unc).any():
            events['xcentroid_err'] = 0.5
            events['ycentroid_err'] = 0.5
        else:
            events['xcentroid_err'] = np.sqrt(events['xcentroid_err']**2 + wcs_unc[0]**2) * 1.136
            events['ycentroid_err'] = np.sqrt(events['ycentroid_err']**2 + wcs_unc[1]**2) * 1.136

        # -- Remove all events with single frame durations -- #
        # fake_events = events[(events.frame_duration==1)&(events.frame_bin==1)].copy()
        # fake_events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/single_frame_events.csv')

        # real_events = events[(events.frame_duration>1)|(events.frame_bin>1)].copy()
        # real_events['total_events'] = real_events.groupby('objid')['objid'].transform('size')
        # real_events['eventid'] = real_events.groupby('objid').cumcount() + 1

        events['total_events'] = events.groupby('objid')['objid'].transform('size')
        events['eventid'] = events.groupby('objid').cumcount() + 1

        self.events = events 

    def _events_physical_units(self):
        """
        Give physical values to event.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        events = deepcopy(self.events)

        events['ra'],events['dec'] = self.wcs.all_pix2world(events['xcentroid'],events['ycentroid'],0)

        delta = 0.1
        # Derivatives w.r.t x
        ra_px, dec_px = self.wcs.all_pix2world(events['xcentroid'] + delta, events['ycentroid'], 0)
        dra_dx = (ra_px - events['ra']) / delta
        ddec_dx = (dec_px - events['dec']) / delta
        
        # Derivatives w.r.t y
        ra_py, dec_py = self.wcs.all_pix2world(events['xcentroid'],  events['ycentroid'] + delta, 0)
        dra_dy = (ra_py - events['ra']) / delta
        ddec_dy = (dec_py - events['dec']) / delta
        
        events['ra_err'] = np.sqrt((dra_dx * events['xcentroid_err'])**2 + (dra_dy * events['ycentroid_err'])**2)
        events['dec_err'] = np.sqrt((ddec_dx * events['xcentroid_err'])**2 + (ddec_dy * events['ycentroid_err'])**2)

        events['ra_err'] = np.abs(events['ra_err'] * np.cos(np.radians(events['dec'])))   # account for cos(dec) factor in RA

        coords = SkyCoord(ra=events['ra'].values*u.degree,dec=events['dec'].values*u.degree)
        events['gal_l'] = coords.galactic.l.value
        events['gal_b'] = coords.galactic.b.value

        startframes = events['frame_start'] * events['frame_bin']
        startframes[startframes>=len(self.time)] = len(self.time)-1
        events['mjd_start'] = self.time[startframes]

        endframes = events['frame_end'] * events['frame_bin']
        endframes[endframes>=len(self.time)] = len(self.time)-1
        events['mjd_end'] = self.time[endframes]

        maxframes = events['frame_max'] * events['frame_bin']
        maxframes[maxframes>=len(self.time)] = len(self.time)-1
        
        events['mjd_max'] = self.time[maxframes]

        events['mjd_duration'] = events['mjd_end'] - events['mjd_start']

        events['mag_min'] = -2.5*np.log10(events['flux_max'])

        self.events = events

    def _flag_asteroids(self):
        """
        Flag asteroids in events.
        """

        from .tools import Frame_Bin
        from .asteroid_detection import Threshold_Asteroid_Checker, Straight_Line_Asteroid_Checker, Tag_Asteroids

        events = deepcopy(self.events)

        # -- Correct light curves of asteroids whose locations change -- #
        frame_bins = np.unique(events.frame_bin)
        events_list = []
        for frame_bin in frame_bins:
            evs = events[events.frame_bin == frame_bin]
            time,flux = Frame_Bin(self.sector,self.cam,self.time,self.flux,frame_bin)
            evs = Threshold_Asteroid_Checker(time,flux,evs)        # Generally best, checks for centre of mass movement and light curve Gaussianity 
            evs = Straight_Line_Asteroid_Checker(time,flux,evs)    # Picks up events with weirdly long event boundaries  
            evs = _Recheck_asteroid_lcs(time,flux,evs)              # Correct light curves of asteroids whose locations change -- #
            events_list.append(evs)
        
        evs = pd.concat(events_list)

        evs = Tag_Asteroids(evs,spatial_lim = self.flux.shape[1])

        evs = _Flag_Event_Frames(evs,len(self.time)-1,std=7)    # Now that all lcs / frame_maxes are finalised, we can flag frames with too many events in them

        self.events = evs

    def _flag_known_asteroids(self):
        """
        Cross-match detected sources against this cut's predicted asteroid
        ephemeris (predict_asteroids(), from asteroid_prediction.py), if
        one exists -- grounds classification in a real, orbit-propagated
        identity rather than only the drift-shape heuristic _flag_asteroids
        uses. A no-op (skipped) if predict_asteroids hasn't been run for
        this cut, so this stays optional rather than a hard dependency.

        Simple closest-approach proximity match (identify_known_asteroids),
        not a motion/offset fit: TESSELLATE's own per-frame detected
        centroids aren't reliable enough for a multi-frame residual check
        to be trustworthy, so a single closest-approach distance is used
        instead. The predicted-vs-measured offset applied before that
        check is loaded from asteroid_lightcurves()'s precomputed
        shift-and-stack centroid measurement (_AsteroidCutOffset, far more
        precise than anything derivable from the detected centroids here)
        when available, falling back to identify_known_asteroids' own
        detected-position pooling otherwise.

        Matched events are upgraded to classification='Asteroid' even if
        _flag_asteroids' drift heuristic missed them (a real orbital match
        is stronger evidence than centroid-drift shape alone); asteroid_id
        (Tag_Asteroids' internal integer track group) is left untouched --
        the real MPC designation goes in new known_asteroid_* columns
        instead, so downstream code relying on asteroid_id being an int
        keeps working unchanged.
        """
        from .tools import load_table, table_exists
        from .asteroid_photometry import identify_known_asteroids

        ephemeris_path = f'{self.path}/Cut{self.cut}of{self.n**2}/asteroids/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{self.cut}_of{self.n**2}_Asteroids.csv'
        if not table_exists(ephemeris_path) or self.sources is None or len(self.sources) == 0:
            self.events['known_asteroid_designation'] = None
            self.events['known_asteroid_dist_px'] = np.nan
            self.events['known_asteroid_frame'] = None
            return

        ephemeris = load_table(ephemeris_path)
        if len(ephemeris) == 0:
            self.events['known_asteroid_designation'] = None
            self.events['known_asteroid_dist_px'] = np.nan
            self.events['known_asteroid_frame'] = None
            return

        offset_path = f'{self.path}/Cut{self.cut}of{self.n**2}/asteroids/sector{self.sector}_cam{self.cam}_ccd{self.ccd}_cut{self.cut}_of{self.n**2}_AsteroidCutOffset.csv'
        offset_x = offset_y = None
        if table_exists(offset_path):
            offset_row = load_table(offset_path)
            if len(offset_row):
                offset_x = offset_row['offset_x'].iloc[0]
                offset_y = offset_row['offset_y'].iloc[0]

        matches = identify_known_asteroids(self.sources, ephemeris, offset_x=offset_x, offset_y=offset_y)
        matches = matches.rename(columns={
            'designation': 'known_asteroid_designation',
            'dist_px': 'known_asteroid_dist_px',
            'frame': 'known_asteroid_frame'})[
            ['objid','known_asteroid_designation','known_asteroid_dist_px',
             'known_asteroid_frame','matched_known_asteroid']]

        self.events = self.events.merge(matches,on='objid',how='left')
        matched = self.events['matched_known_asteroid'].fillna(False)
        self.events.loc[matched,'classification'] = 'Asteroid'
        self.events = self.events.drop(columns=['matched_known_asteroid'])

    def _catalogue_crossmatch(self,sigma=3):
        """
        Crossmatch events with stars / variables.
        """
        
        events = deepcopy(self.events)

        # -- Cross matches location to Gaia -- #
        gaia = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/local_gaia_cat.csv')
        events['gaia_id'] = '-'
        for i,event in events.iterrows():
            if event.classification not in ['Asteroid','CosmicRay','Junk']:

                # --- RA wraparound-safe box pre-filter ---
                dra_raw = gaia.ra - event.ra
                dra_wrapped = (dra_raw + 180) % 360 - 180

                box_mask = (
                    (np.abs(dra_wrapped * np.cos(np.radians(event.dec))) < 3 * event.ra_err) &
                    (np.abs(gaia.dec - event.dec) < 3 * event.dec_err)
                )

                if box_mask.any():
                    
                    sub_dra_wrapped = dra_wrapped[box_mask]
                    sub_source = gaia.Source[box_mask].values
                    sub_rpmag = gaia.RPmag[box_mask].values
                    sub_gmag = gaia.Gmag[box_mask].values
                    sub_dec = gaia.dec[box_mask].values

                    dra_arcsec = sub_dra_wrapped * np.cos(np.radians(event.dec)) * 3600
                    ddec_arcsec = (sub_dec - event.dec) * 3600

                    ra_err_arcsec = event.ra_err * 3600
                    dec_err_arcsec = event.dec_err * 3600

                    mahalanobis = np.sqrt((dra_arcsec / ra_err_arcsec)**2 + (ddec_arcsec / dec_err_arcsec)**2)
                    rad_mask = mahalanobis <= 3

                    if rad_mask.any():

                        cand_source = sub_source[rad_mask]
                        cand_rpmag = sub_rpmag[rad_mask]
                        cand_gmag = sub_gmag[rad_mask]

                        valid_rp = ~np.isnan(cand_rpmag)
                        valid_g = ~np.isnan(cand_gmag)

                        if valid_rp.any():
                            best_idx = np.nanargmin(np.where(valid_rp, cand_rpmag, np.inf))
                        elif valid_g.any():
                            best_idx = np.nanargmin(np.where(valid_g, cand_gmag, np.inf))
                        else:
                            best_idx = 0

                        events.loc[i,'gaia_id'] = str(cand_source[best_idx])

                                                    # inside = gaia[(abs(gaia.ra-event.ra) < sigma*event.ra_err)&
                                                    #             (abs(gaia.dec-event.dec) < sigma*event.dec_err)]
                                                    # if len(inside) > 0:
                                                    #     valid_rp = inside.dropna(subset=['RPmag'])
                                                    #     valid_g = inside.dropna(subset=['Gmag'])
                                                    #     if len(valid_rp) > 0:
                                                    #         best_idx = valid_rp.RPmag.idxmin()
                                                    #     elif len(valid_g) > 0:
                                                    #         best_idx = valid_g.Gmag.idxmin()
                                                    #     else:
                                                    #         best_idx = inside.index[0]

                                                    #     events.loc[i, 'gaia_id'] = gaia.at[best_idx, 'Source'] 
        
        # -- Cross matches location to variable catalog -- #
        variables = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/variable_catalog.csv')
        for i,event in events.iterrows():
            if event.classification not in ['Asteroid','CosmicRay','Junk']:
                inside = variables[(abs(variables.ra-event.ra) < sigma*event.ra_err)&
                            (abs(variables.dec-event.dec) < sigma*event.dec_err)]
                if len(inside) > 0:
                    events.loc[i,'classification'] = inside.iloc[0].Type

        self.events = events

    # def _catalogue_crossmatch(self, sigma=3):
    #     """Crossmatch events with stars / variables."""
    #     events = deepcopy(self.events)

    #     # Pre-filter to non-junk events so we only work on relevant rows
    #     skip = {'Asteroid', 'CosmicRay', 'Junk'}
    #     valid_mask = ~events.classification.isin(skip)
    #     valid_idx  = events.index[valid_mask]
    #     ev = events.loc[valid_idx]

    #     ev_ra      = ev.ra.values
    #     ev_dec     = ev.dec.values
    #     ra_tol     = sigma * ev.ra_err.values    # shape (N_ev,)
    #     dec_tol    = sigma * ev.dec_err.values

    #     # ------------------------------------------------------------------ #
    #     # Gaia crossmatch                                                      #
    #     # ------------------------------------------------------------------ #
    #     gaia = pd.read_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/local_gaia_cat.csv')
    #     events['gaia_id'] = '-'

    #     g_ra  = gaia.ra.values
    #     g_dec = gaia.dec.values
    #     g_rp  = gaia.RPmag.values
    #     g_g   = gaia.Gmag.values
    #     g_src = gaia.Source.values

    #     # Broadcasting: (N_ev, N_gaia) — all pairwise abs-differences at once
    #     ra_ok  = np.abs(g_ra[np.newaxis, :]  - ev_ra[:, np.newaxis])  < ra_tol[:, np.newaxis]
    #     dec_ok = np.abs(g_dec[np.newaxis, :] - ev_dec[:, np.newaxis]) < dec_tol[:, np.newaxis]
    #     match_matrix = ra_ok & dec_ok   # (N_ev, N_gaia) bool

    #     for j, orig_i in enumerate(valid_idx):
    #         cols = np.where(match_matrix[j])[0]
    #         if len(cols) == 0:
    #             continue
    #         rp_sub = g_rp[cols]
    #         g_sub  = g_g[cols]
    #         # Priority: brightest RPmag → brightest Gmag → first match
    #         rp_ok = ~np.isnan(rp_sub)
    #         g_ok  = ~np.isnan(g_sub)
    #         if rp_ok.any():
    #             best = cols[np.nanargmin(rp_sub)]
    #         elif g_ok.any():
    #             best = cols[np.nanargmin(g_sub)]
    #         else:
    #             best = cols[0]
    #         events.at[orig_i, 'gaia_id'] = int(g_src[best])

    #     # ------------------------------------------------------------------ #
    #     # Variable catalogue crossmatch                                        #
    #     # ------------------------------------------------------------------ #
    #     variables = pd.read_csv(
    #         f'{self.path}/Cut{self.cut}of{self.n**2}/variable_catalog.csv'
    #     )

    #     v_ra   = variables.ra.values
    #     v_dec  = variables.dec.values
    #     v_type = variables.Type.values

    #     ra_ok  = np.abs(v_ra[np.newaxis, :]  - ev_ra[:, np.newaxis])  < ra_tol[:, np.newaxis]
    #     dec_ok = np.abs(v_dec[np.newaxis, :] - ev_dec[:, np.newaxis]) < dec_tol[:, np.newaxis]
    #     match_matrix = ra_ok & dec_ok   # (N_ev, N_var)

    #     # np.argmax on bool rows finds the first True index in O(N_ev × N_var)
    #     has_match = match_matrix.any(axis=1)             # (N_ev,) bool
    #     first_col = match_matrix.argmax(axis=1)          # (N_ev,) — valid only where has_match

    #     hit_positions = np.where(has_match)[0]
    #     for j in hit_positions:
    #         events.at[valid_idx[j], 'classification'] = v_type[first_col[j]]

    #     self.events = events


    def _crossmatch_framebin(self):
        """
        Crossmatch between time_bins — parallelised over asteroid_crossbin_group.
        """
        from collections import Counter
        from joblib import Parallel, delayed
        from tqdm import tqdm

        events = deepcopy(self.events)

        events['crossbin_ids'] = [[] for _ in range(len(events))]

        # -- Group spatially -- #
        events = _Spatial_group(events, colname='crossbin_group',          distance=1, min_samples=1)
        events = _Spatial_group(events, colname='asteroid_crossbin_group', distance=2, min_samples=1)

        # -- Split into independent chunks by asteroid_crossbin_group -- #
        unique_ag = np.unique(events['asteroid_crossbin_group'])
        chunks = {
            ag: events[events['asteroid_crossbin_group'] == ag]
            for ag in unique_ag
        }

        results = Parallel(n_jobs=-1)(
            delayed(_Crossbin_group_worker)(ag, df)
            for ag, df in tqdm(chunks.items(),desc='Bin crossmatching')
        )

        # -- Merge crossbin_ids back into events -- #
        # Build the final dict in pure Python first, then assign once
        ts = clock()
        merged = {}
        for result_df in results:
            merged.update(result_df['crossbin_ids'].to_dict())
        print(f'       merged to dict ({clock()-ts:.0f}s)',flush=True)


        events['crossbin_ids'] = [merged.get(idx, []) for idx in events.index]
        events = events.drop(columns=['crossbin_group', 'asteroid_crossbin_group'])


        # -- Find which indices are referenced more than once -- #
        ts = clock()
        all_ids = [i for ids in events['crossbin_ids'] for i in ids]
        counts = Counter(all_ids)
        all_referenced = frozenset(i for i, c in counts.items() if c > 1)
        print(f'       found indices ({clock()-ts:.0f}s)',flush=True)

        # -- Clear solo crossbin_ids -- #
        # .map() has less per-row overhead than .apply()
        events['crossbin_ids'] = events['crossbin_ids'].map(
            lambda ids: ids if not ids or all_referenced.isdisjoint(ids) else ids
        )
        # Clear rows where none of their ids survived
        events['crossbin_ids'] = events['crossbin_ids'].map(
            lambda ids: [] if all_referenced.isdisjoint(ids) else ids
        )

        # -- Remap crossbin_ids -- #
        id_map = {old: new for new, old in enumerate(sorted(all_referenced))}
        events['crossbin_ids'] = events['crossbin_ids'].map(
            lambda x: [id_map[i] for i in x if i in id_map]
        )


        # # -- Merge crossbin_ids back into events -- #
        # events['crossbin_ids'] = [[] for _ in range(len(events))]
        # for result_df in results:
        #     for idx, ids in result_df['crossbin_ids'].items():
        #         events.at[idx, 'crossbin_ids'] = ids

        # events = events.drop(columns=['crossbin_group', 'asteroid_crossbin_group'])

        # # -- Find which indices are referenced more than once -- #
        # all_ids = [i for ids in events['crossbin_ids'] for i in ids]
        # counts = Counter(all_ids)
        # all_referenced = set(i for i, c in counts.items() if c > 1)

        # # -- Clear solo crossbin_ids -- #
        # events['crossbin_ids'] = events.apply(
        #     lambda row: [] if not any(i in all_referenced for i in row['crossbin_ids'])
        #     else row['crossbin_ids'],
        #     axis=1
        # )

        # # -- Remap crossbin_ids -- #
        # id_map = {old: new for new, old in enumerate(sorted(all_referenced))}
        # events['crossbin_ids'] = events['crossbin_ids'].apply(
        #     lambda x: [id_map[i] for i in x if i in id_map]
        # )

        self.events = events
    

    # def _crossmatch_framebin(self):
    #     """
    #     Crossmatch between time_bins.
    #     """
        
    #     from collections import Counter

    #     events = deepcopy(self.events)
    #     events['crossbin_ids'] = [[] for _ in range(len(events))]

    #     # -- Group spatially -- #
    #     events = _Spatial_group(events, colname='crossbin_group', distance=1, min_samples=1)
    #     events = _Spatial_group(events, colname='asteroid_crossbin_group', distance=2, min_samples=1)

    #     # -- Iterate through spatial groups -- # 
    #     for group in np.unique(events.crossbin_group):
    #         group_df = events[events.crossbin_group == group].sort_values('frame_bin')

    #         for i, row in group_df.iterrows():
    #             # If this event has no ID yet, assign its own
    #             if not events.at[i, 'crossbin_ids']:
    #                 events.at[i, 'crossbin_ids'] = [i]

    #             current_ids = events.at[i, 'crossbin_ids']

    #             # For asteroids, use the wider spatial group to find coarser bin candidates
    #             if row['classification'] == 'Asteroid':
    #                 asteroid_group = events.at[i, 'asteroid_crossbin_group']
    #                 coarser = events[
    #                     (events['asteroid_crossbin_group'] == asteroid_group) &
    #                     (events['frame_bin'] > row['frame_bin'])
    #                 ].sort_values('frame_bin')
    #             else:
    #                 coarser = group_df[group_df['frame_bin'] > row['frame_bin']]

    #             for j, upper in coarser.iterrows():
    #                 consistent = (
    #                     (row['frame_start'] * row['frame_bin'] <= upper['frame_end'] * upper['frame_bin'] + upper['frame_bin']) &
    #                     (row['frame_end'] * row['frame_bin'] >= upper['frame_start'] * upper['frame_bin'] - upper['frame_bin'])
    #                 )
    #                 if consistent:
    #                     # Merge IDs, avoiding duplicates
    #                     merged = list(set(events.at[j, 'crossbin_ids'] + current_ids))
    #                     events.at[j, 'crossbin_ids'] = merged

    #     events = events.drop(columns=['crossbin_group', 'asteroid_crossbin_group'])

    #     # -- Find which indices are present more than once -- #
    #     all_ids = [i for ids in events['crossbin_ids'] for i in ids]
    #     counts = Counter(all_ids)
    #     all_referenced = set(i for i, c in counts.items() if c > 1)

    #     # -- Clear solo crossbin_ids -- #
    #     events['crossbin_ids'] = events.apply(
    #         lambda row: [] if not any(i in all_referenced for i in row['crossbin_ids']) else row['crossbin_ids'], axis=1
    #     )

    #     # -- Remap crossbin_ids -- #
    #     id_map = {old: new for new, old in enumerate(sorted(all_referenced))}
    #     events['crossbin_ids'] = events['crossbin_ids'].apply(
    #         lambda x: [id_map[i] for i in x if i in id_map]
    #     )

    #     self.events = events
 
    def _order_events_columns(self):
        """
        Order events columns.
        """

        ordered_cols = [
            # Primary Identification
            'objid', 'eventid', 'classification',
            'sector', 'camera', 'ccd', 'cut',

            # Centroid Positions
            'xcentroid', 'ycentroid',
            'xcentroid_err','ycentroid_err',
            'xint', 'yint','xccd', 'yccd',
            'xcentroid_det', 'ycentroid_det', 
            'xcentroid_psf', 'ycentroid_psf',
            'xcentroid_err_psf','ycentroid_err_psf',

            # Astrometry
            'ra', 'dec','ra_err','dec_err',
            'gal_l', 'gal_b',

            # Time and Frame Info
            'frame_start', 'frame_end','frame_max','frame_duration',
            'mjd_start', 'mjd_end', 'mjd_max', 'mjd_duration',

            # Photometry
            'flux_max', 'mag_min','flux_sign',
            'image_sig_max', 'lc_sig_max', 'lc_sig_med','lc_flat',
            'snr_psf',

            # Morphology
            'psf_like', 'psf_diff','psf_stacked','com_motion','gaussian_score',
            'ellipticity','fwhm','neg_extent','bad_frame_flag',

            # # Frequency Domain
            # 'peak_freq', 'peak_power',

            # Secondary Identification
            'source_mask', 'gaia_id', 'crossbin_ids','asteroid_id', # 'prob', 'GaaID', 'cf_class', 'cf_prob',
            'known_asteroid_designation','known_asteroid_dist_px','known_asteroid_frame',

            # Miscellaneous
            'n_detections','total_events','frame_bin', 'TSS Catalogue'
        ]

        # Safely apply ordering
        self.events = self.events[[col for col in ordered_cols if col in self.events.columns]]

    def _TSS_catalogue_names(self):
        """
        Generate a TSS style catalogue name for each event.
        """

        from astropy.coordinates import SkyCoord
        import astropy.units as u

        tss_names = []
        for _,event in self.events.iterrows():
            c = SkyCoord(ra=event['ra'] * u.deg, dec=event['dec'] * u.deg)

            ra_hms = c.ra.hms
            dec_dms = c.dec.dms

            RAh = f'{int(ra_hms.h):02d}'
            RAm = f'{int(ra_hms.m):02d}'
            RAs = f'{ra_hms.s:05.2f}'

            sign = '+' if dec_dms.d >= 0 else '-'
            DECd = f'{abs(int(dec_dms.d)):02d}'
            DECm = f'{abs(int(dec_dms.m)):02d}'
            DECs = f'{abs(dec_dms.s):05.2f}'

            tss_name = f'TSS {RAh}{RAm}{RAs}{sign}{DECd}{DECm}{DECs}'
            if event['total_events']==1:
                tss_name += f"T{int(event['mjd_start'])}"
            tss_names.append(tss_name)
        self.events['TSS Catalogue'] = tss_names


    def find_events(self):

        # -- Group these sources into unique objects based on the objid -- #
        ts = clock()
        self._get_all_independent_events()
        print(f'   Separated into individual events -- done! ({(clock()-ts):.0f}s)',flush=True)

        # -- Get physical units for events -- #
        self._events_physical_units()
        print(f'   Getting time/coords/flux information -- done!',flush=True)

        # -- Tag asteroids -- #
        ts = clock()
        self._flag_asteroids()
        print(f'   Checking for asteroids -- done! ({(clock()-ts):.0f}s)',flush=True)

        # -- Cross-match against predicted (orbit-propagated) asteroids, if available -- #
        ts = clock()
        self._flag_known_asteroids()
        print(f'   Checking against predicted asteroids -- done! ({(clock()-ts):.0f}s)',flush=True)

        self.events = self.events.drop_duplicates(subset=['frame_bin','xint','yint','frame_max'],keep='first')

        # -- Crossmatch with catalogues -- #
        ts = clock()
        self._catalogue_crossmatch()
        print(f'   Crossmatching with Gaia and Variables -- done! ({(clock()-ts):.0f}s)',flush=True)

        # -- Crossmatch between different frame_bins -- #
        ts = clock()
        print(f'   Crossmatching between time bins',flush=True)
        self._crossmatch_framebin()
        print(f'   Crossmatching between time bins -- done! ({(clock()-ts):.0f}s)',flush=True)

        # -- Get TSS Catalogue Names -- #        
        self._TSS_catalogue_names()
        print(f'   Getting TSS Catalogue Names -- done!',flush=True)

        # -- Order nicely -- #
        self._order_events_columns()  

        # -- Save out results to csv file -- #
        self.events.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/{self._inj_path}/detected_events.csv',index=False)
        


    # ------------------------------ Object finding function ------------------------------ #

    def find_objects(self):
        """
        Extract spatial objects.
        """

        objids = self.events['objid'].unique()

        columns = [
            'frame_bin','objid', 'sector', 'cam', 'ccd', 'cut', 'xcentroid', 'ycentroid', 
            'ra', 'dec', 'gal_l', 'gal_b', 'xcentroid_err','ycentroid_err','ra_err','dec_err',
            'lc_sig_max', 'flux_maxsig', 'frame_maxsig',
            'mjd_maxsig','psf_maxsig','flux_sign', 'n_events',
            'min_eventlength_frame', 'max_eventlength_frame',
            'min_eventlength_mjd','max_eventlength_mjd','gaia_id','classification','TSS Catalogue',
            'known_asteroid_designation','known_asteroid_dist_px','known_asteroid_frame'
        ]
        objects = pd.DataFrame(columns=columns)

        for objid in objids:
            obj = self.events[self.events['objid'] == objid]

            maxevent = obj.iloc[obj['image_sig_max'].argmax()]

            if maxevent['classification'] == 'Asteroid' and len(obj) < 2:
                classification = 'Asteroid'
                asteroid_id = maxevent['asteroid_id']
            else:
                classification = obj['classification'].mode()[0]
                asteroid_id = '-'

            if classification == 'RRLyrae':
                classification = 'VRRLyr'

            row_data = {
                'objid': objid,
                'classification': classification,    
                'sector': maxevent['sector'],
                'cam': maxevent['camera'],
                'ccd': maxevent['ccd'],
                'cut': maxevent['cut'],         
                'xcentroid': maxevent['xcentroid'],
                'ycentroid': maxevent['ycentroid'],
                'ra': maxevent['ra'],
                'dec': maxevent['dec'],
                'gal_l': maxevent['gal_l'],
                'gal_b': maxevent['gal_b'],
                'xcentroid_err': maxevent['xcentroid_err'],
                'ycentroid_err': maxevent['ycentroid_err'],
                'ra_err': maxevent['ra_err'],
                'dec_err': maxevent['dec_err'],
                'image_sig_max': maxevent['image_sig_max'],
                'lc_sig_max': maxevent['lc_sig_max'],
                'flux_maxsig': maxevent['flux_max'],
                'frame_maxsig': maxevent['frame_max'],
                'mjd_maxsig': maxevent['mjd_max'],
                'psf_maxsig': maxevent['psf_like'],
                'min_eventlength_frame': obj['frame_duration'].min(),
                'max_eventlength_frame': obj['frame_duration'].max(),
                'min_eventlength_mjd': obj['mjd_duration'].min(),
                'max_eventlength_mjd': obj['mjd_duration'].max(),
                'gaia_id' : maxevent['gaia_id'],
                'asteroid_id' : asteroid_id,
                'frame_bin':  maxevent['frame_bin'],
                'flux_sign': np.sum(obj['flux_sign'].unique()).astype(int),
                'n_events': len(obj),
                'TSS Catalogue' : maxevent['TSS Catalogue'],
                'known_asteroid_designation': maxevent.get('known_asteroid_designation'),
                'known_asteroid_dist_px': maxevent.get('known_asteroid_dist_px'),
                'known_asteroid_frame': maxevent.get('known_asteroid_frame'),
            }

            obj_row = pd.DataFrame([row_data])
            objects = pd.concat([objects, obj_row], ignore_index=True)

        self.objects = objects

        self.objects.to_csv(f'{self.path}/Cut{self.cut}of{self.n**2}/{self._inj_path}/detected_objects.csv',index=False)
        
        
    # ------------------------------ Main search function ------------------------------ #

    def transient_search(self,cut,mode='starfind',prf_path='/fred/oz335/_local_TESS_PRFs',time_bins=['10min'],min_snr=3.0):

        import os

        # -- Check if using starfinder and/or sourcedetect for detection -- #
        self.mode = mode

        if self.sector < 4:
            self.prf_path = f'{prf_path}/Sectors1_2_3'
        else:
            self.prf_path = f'{prf_path}/Sectors4+'
        
        # -- Gather time/flux data for the cut -- #
        print('-------Preloading sources / events-------',flush=True)
        if cut != self.cut:
            self.gather_data(cut,bkg=True,mask=True,ref=True)

        # -- Preload self.sources and self.events if they're already made, self.objects can't be made otherwise this function wouldn't be called -- #
        self.gather_results(cut=cut)
        print('\n') 

        if self.sources is None:
            print('-------Source finding (see progress in errors log file)-------',flush=True)
            self.find_sources(time_bins,min_snr=min_snr)
            print('\n')

        if not os.path.exists(f'{self.prf_path}/cam{self.cam}_ccd{self.ccd}/snr_to_localisation/cut{cut}of{int(self.n**2)}_coeffs_y.npy'):
            from .localisation import simulate_cut_psf_fitting

            print('-------Simulating PSF fit accuracy (see progress in errors log file)-------',flush=True)
            simulate_cut_psf_fitting(self.prf_path,self.sector,self.cam,self.ccd,cut,n=self.n,nfits=100000,nMedians=10)
            print('\n')

        # -- self.events contains all individual events, grouped by time and space -- #  
        if self.events is None:
            print('-------Event finding (see progress in errors log file)-------',flush=True)
            self.find_events()
            print('\n')

        # -- self.objects contains all individual spatial objects -- #  
        print('-------Object finding-------',flush=True)
        self.find_objects() 