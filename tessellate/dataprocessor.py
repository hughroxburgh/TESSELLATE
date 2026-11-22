import os
import numpy as np
from time import time as t

from .tools import _Save_space, _Remove_emptys, _Extract_fits, _Print_buff, save_compact_array, save_table, load_table, table_exists
    
# def _get_wcs(path,wcs_path_check,verbose=1):
#     """
#     Get WCS data from a file in the path
#     """

#     import shutil
#     from glob import glob
#     from astropy.wcs import WCS

#     if os.path.exists(wcs_path_check):
#         wcsFile = _Extract_fits(wcs_path_check)
#         wcsItem = WCS(wcsFile[1].header)
#     else:
#         if glob(f'{path}/*ffic.fits'):
#             done = False
#             i = 0
#             while not done:
#                 filepath = glob(f'{path}/*ffic.fits')[i]
#                 file = _Extract_fits(filepath)
#                 wcsItem = WCS(file[1].header)
#                 file.close()
#                 if wcsItem.get_axis_types()[0]['coordinate_type'] == 'celestial':
#                     done = True
#                     shutil.copy2(filepath,wcs_path_check)
#                 else:
#                     i += 1
#         else:
#             if verbose>0:
#                 print('No Data!')
#             return

#     return wcsItem

def _wcs_ok(wcsItem):
    """True if a WCS looks like a real TESS CCD solution: celestial, finite across the chip,
    ~21"/px at the centre, and pixel -> sky -> pixel round trips at the corners."""
    try:
        if not wcsItem.has_celestial:
            return False
        px = np.array([[1068,1024],[1069,1024],[44,0],[2091,0],[44,2047],[2091,2047]],dtype=float)
        sky = wcsItem.all_pix2world(px,0)
        if not np.all(np.isfinite(sky)):
            return False
        ra, dec = np.radians(sky[:2,0]), np.radians(sky[:2,1])
        cos_sep = np.sin(dec[0])*np.sin(dec[1]) + np.cos(dec[0])*np.cos(dec[1])*np.cos(ra[0]-ra[1])
        scale = np.degrees(np.arccos(np.clip(cos_sep,-1,1)))*3600
        if not 15 < scale < 30:
            return False
        back = wcsItem.all_world2pix(sky[2:],0)
        return bool(np.all(np.isfinite(back)) and np.max(np.abs(back - px[2:])) < 1)
    except Exception:
        return False

def find_good_wcs(sector,data_path,cam,ccd,max_downloads=10,verbose=0):
    """
    WCS for a cam/ccd that passes _wcs_ok: the corrected reference if there is one, else the
    already-downloaded FFIs, else single FFIs downloaded spread through the sector (up to
    max_downloads). Returns None if none works.
    """

    from glob import glob
    from astropy.io import fits
    from astropy.wcs import WCS

    path = f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}'

    def read(file):
        try:
            with fits.open(file) as hdul:
                w = WCS(hdul[1].header)
            return w if _wcs_ok(w) else None
        except Exception:
            return None

    if os.path.exists(f'{path}/wcs/ref/corrected.fits'):
        w = read(f'{path}/wcs/ref/corrected.fits')
        if w is not None:
            return w

    tried = set()
    for file in sorted(glob(f'{path}/image_files/*ffic.fits')):
        tried.add(file)
        w = read(file)
        if w is not None:
            return w
        if verbose > 0:
            print(f'Bad WCS in {os.path.basename(file)}, trying another FFI')

    data_processor = DataProcessor(sector=sector,data_path=data_path,verbose=0)
    for ind in range(0,max_downloads*100,100):     # every 100th FFI, so one bad stretch isn't hit repeatedly
        try:
            data_processor.download(cam=cam,ccd=ccd,number=None,single=ind)
        except IndexError:      # past the end of the sector
            break
        for file in sorted(set(glob(f'{path}/image_files/*ffic.fits')) - tried):
            tried.add(file)
            w = read(file)
            if w is not None:
                return w
            if verbose > 0:
                print(f'Bad WCS in {os.path.basename(file)}, trying another FFI')

    if verbose > 0:
        print(f'No FFI with a good WCS found for Sector {sector} Cam {cam} CCD {ccd}')
    return None

CUT_MARGIN_PX = 5      # rows kept beyond each cut's pixel square, for position offsets at its edge
_CUT_WRITE = None      # (processor, CCD ephemeris, per-cut jobs), reaching forked workers without pickling


def _ccd_ephemeris_for_cuts(ephemeris, mjd, cutSize):
    """The CCD-level ephemeris prepared for splitting into cuts: sorted once by (designation, frame),
    so every cut's rows come out already in that order, with designations categorical so forked
    workers never touch (and copy-on-write) millions of Python string objects."""
    import pandas as pd
    df = ephemeris.reset_index(drop=True)
    designation = pd.Categorical(df['designation'])      # categories sorted, as the old string sort
    order = np.lexsort((df['frame'].to_numpy(), designation.codes))
    df = df.iloc[order].reset_index(drop=True)
    df['designation'] = designation[order]
    return dict(df=df, x=df['x'].to_numpy(), y=df['y'].to_numpy(), frame=df['frame'].to_numpy().astype(np.int64),
                mjd=mjd, size=cutSize)


def _cut_ephemeris(ccd, corner, cmjd):
    """One cut's rows of the CCD ephemeris: inside its pixel square plus CUT_MARGIN_PX, at the cut's
    own frame times, with 'frame' re-indexed to the cut's frame order."""
    x0, y0 = corner
    lo_x, hi_x = x0 - CUT_MARGIN_PX, x0 + 2 * ccd['size'] + CUT_MARGIN_PX
    lo_y, hi_y = y0 - CUT_MARGIN_PX, y0 + 2 * ccd['size'] + CUT_MARGIN_PX
    # 'frame' indexes the union of the cuts' times; the cut's own times are members of it bit for bit
    # (np.unique keeps them), so an exact lookup maps union frames to cut frames, -1 elsewhere
    lut = np.full(len(ccd['mjd']), -1, dtype=np.int64)
    lut[np.searchsorted(ccd['mjd'], cmjd)] = np.arange(len(cmjd))
    cut_frame = lut[ccd['frame']]
    x, y = ccd['x'], ccd['y']
    keep = np.flatnonzero((x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y) & (cut_frame >= 0))
    eph = ccd['df'].iloc[keep].reset_index(drop=True)
    eph['frame'] = cut_frame[keep]
    eph['designation'] = eph['designation'].astype(str)
    return eph


def _write_one_cut(k):
    processor, ccd, jobs = _CUT_WRITE
    corner, cmjd, cutFolder, base, plot_trails = jobs[k]
    processor._write_cut_asteroids(_cut_ephemeris(ccd, corner, cmjd), cutFolder, base, corner, ccd['size'],
                                   plot_trails)
    return k


def _write_cuts_parallel(processor, ccd, jobs):
    """Split the CCD ephemeris into cuts and write them in a forked pool sized to the job's CPUs.
    ProcessPoolExecutor, not multiprocessing.Pool: a worker killed (e.g. for memory) raises
    BrokenProcessPool instead of leaving the job waiting forever on the rest."""
    global _CUT_WRITE
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .asteroid_prediction import _available_cpus
    _CUT_WRITE = (processor, ccd, jobs)
    n_workers = max(1, min(len(jobs), _available_cpus()))
    try:
        if n_workers == 1:
            for k in range(len(jobs)):
                _write_one_cut(k)
        else:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork")) as pool:
                list(pool.map(_write_one_cut, range(len(jobs))))
    finally:
        _CUT_WRITE = None


def _cut_properties(wcsItem,n,overlap): 

        intervals = 2048/n

        cutCornersX = [44 + i*intervals for i in range(n)]
        cutCornersY = [i*intervals for i in range(n)]
        cutCorners = np.meshgrid(cutCornersX,cutCornersY)
        cutCorners = np.floor(np.stack((cutCorners[0],cutCorners[1]),axis=2).reshape(n**2,2))

        idx = np.arange(n * n)
        cols = idx % n
        rows = idx // n
        cutCorners[cols == n - 1, 0] -= overlap
        cutCorners[(cols != 0) & (cols != n - 1), 0] -= overlap // 2
        cutCorners[rows == n - 1, 1] -= overlap
        cutCorners[(rows != 0) & (rows != n - 1), 1] -= overlap // 2

        intervals = np.ceil(intervals)
        rad = np.ceil(intervals / 2) + overlap//2

        cutCentrePx = cutCorners + rad
        cutCentreCoords = np.array(wcsItem.all_pix2world(cutCentrePx[:,0],cutCentrePx[:,1],0)).transpose()

        return cutCorners,cutCentrePx,cutCentreCoords,rad

def _parallel_cuts(sector,cam,ccd,cut,n,cube_path,file_path,coords,size,verbose):

    from astrocut import CutoutFactory

    name = f'sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
    if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
        print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
    else:
        if verbose > 0:
            print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
        
        my_cutter = CutoutFactory() # astrocut class
        _Save_space(f'{file_path}/Cut{cut}of{n**2}')

        # -- Cut -- #
        cut_file = my_cutter.cube_cut(cube_path, 
                                        f"{coords[0]} {coords[1]}", 
                                        (size,size), 
                                        output_path = f'{file_path}/Cut{cut}of{n**2}',
                                        target_pixel_file = name,
                                        verbose=(verbose>1)) 

        if verbose > 0:
            print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
            print('\n')

# ---------------------------------------------------------------------------
# Direct cuts: TPFs written straight from the FFIs, no cube
# ---------------------------------------------------------------------------

FRAME_STAT_BLOCKS = 8   # TessFixer ranks reference frames on the medians of an 8x8 grid of 256 px blocks


def _ffi_time(file):
    """Start time (days) from an FFI's tess{YYYYDDDHHMMSS} file name -- for ordering and the part split only."""
    import re
    import datetime
    stamp = re.search(r'tess(\d{13})', os.path.basename(file)).group(1)
    date = datetime.datetime.strptime(stamp, '%Y%j%H%M%S')
    return (date - datetime.datetime(2000, 1, 1)).total_seconds() / 86400


def _frame_stats_path(ccd_folder, sector, cam, ccd):
    return f'{ccd_folder}/sector{sector}_cam{cam}_ccd{ccd}_frame_stats.npz'


def _tpf_row_dtype(size, name_len):
    """One cadence of a direct-cut TPF, big-endian as FITS stores it. lightkurve takes the WCS from APERTURE."""
    return np.dtype([('TIME', '>f8'), ('TIMECORR', '>f4'), ('CADENCENO', '>i4'), ('QUALITY', '>i4'),
                     ('FLUX', '>f4', (size, size)), ('FLUX_ERR', '>f4', (size, size)),
                     ('POS_CORR1', '>f4'), ('POS_CORR2', '>f4'), ('FFI_FILE', f'S{name_len}')])


def _tpf_columns(size, name_len):
    from astropy.io import fits
    dim = f'({size},{size})'
    return [fits.Column('TIME', 'D', unit='BJD - 2457000, days'),
            fits.Column('TIMECORR', 'E', unit='d'),
            fits.Column('CADENCENO', 'J'),
            fits.Column('QUALITY', 'J'),
            fits.Column('FLUX', f'{size*size}E', dim=dim, unit='e-/s'),
            fits.Column('FLUX_ERR', f'{size*size}E', dim=dim, unit='e-/s'),
            fits.Column('POS_CORR1', 'E', unit='pixel'),
            fits.Column('POS_CORR2', 'E', unit='pixel'),
            fits.Column('FFI_FILE', f'{name_len}A')]


def _cut_linear_wcs(wcsItem, x0, y0, size):
    """
    Linear TAN WCS fitted to the full (SIP) FFI WCS across one cut, as astrocut writes in a cut's APERTURE header,
    with the fit's max and rms sky offsets (deg) for its WCS_MSEP / WCS_SIG keywords.
    """
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    g = np.linspace(0, size - 1, 20)
    xx, yy = [a.ravel() for a in np.meshgrid(g, g)]
    sky = SkyCoord(*wcsItem.all_pix2world(xx + x0, yy + y0, 0), unit='deg')
    fit = fit_wcs_from_points((xx, yy), sky, proj_point='center', projection='TAN')
    sep = sky.separation(SkyCoord(*fit.all_pix2world(xx, yy, 0), unit='deg')).deg
    return fit, sep.max(), np.sqrt(np.mean(sep**2))


APERTURE_HEADER_BLOCKS = 12   # fixed APERTURE header size (2880-byte blocks): room for TessFixer's order-6 SIP WCS,
                               # so _stamp_tpf_wcs can rewrite it in place without moving any data


def _fixed_header_bytes(header, nblocks):
    """A FITS header padded with blank records to exactly nblocks blocks, so it can be rewritten in place."""
    cards = header.tostring(sep='', endcard=False, padding=False)
    size = nblocks * 2880
    if len(cards) + 80 > size:
        raise ValueError(f'Header needs {len(cards) // 80 + 1} records, only {size // 80} fit in {nblocks} blocks')
    return (cards + ' ' * (size - 80 - len(cards)) + 'END'.ljust(80)).encode()


def _aperture_header(size, wcs_header, extra):
    """The APERTURE extension's header: image shape + WCS + extra (keyword, (value, comment)) cards."""
    from astropy.io import fits
    header = fits.ImageHDU(np.ones((size, size), np.int32), header=wcs_header, name='APERTURE').header
    for key, value in extra.items():
        header[key] = value
    return header


def _stamp_tpf_wcs(path, wcs_header, extra):
    """Rewrite a direct-cut TPF's APERTURE header (its WCS, which tpf.wcs reads) in place."""
    from astropy.io import fits
    with fits.open(path) as h:
        info = h.fileinfo(2)
        size = h[2].header['NAXIS1']
    nbytes = info['datLoc'] - info['hdrLoc']
    if nbytes != APERTURE_HEADER_BLOCKS * 2880:
        raise ValueError(f'{path} has no fixed-size APERTURE header (not a direct cut?)')
    block = _fixed_header_bytes(_aperture_header(size, wcs_header, extra), APERTURE_HEADER_BLOCKS)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, block, info['hdrLoc'])
    finally:
        os.close(fd)


def _cutwcs_header(cut_wcs, size):
    """
    CutWCS (TessFixer's corrected CCD WCS, shifted by the cut corner and the cut's median source offset) as an
    ordinary SIP WCS header for the cut's pixels. Slicing moves CRPIX by the corner; adding (dx, dy) to CRPIX
    shifts the whole mapping, SIP terms included, exactly as CutWCS.all_world2pix does.
    """
    x0, y0 = int(cut_wcs.corner[0]), int(cut_wcs.corner[1])
    header = cut_wcs.wcs[y0:y0+size, x0:x0+size].to_header(relax=True)
    header['CRPIX1'] += float(cut_wcs.dx)
    header['CRPIX2'] += float(cut_wcs.dy)
    return header


def _create_direct_tpf(path, ffi_hdrs, wcsItem, wcs_ffi, x0, y0, size, nrows, name_len, sector, cam, ccd, cut, n):
    """
    Write a cut TPF's headers and APERTURE extension around an empty (sparse) PIXELS table of nrows cadences,
    for workers to fill row by row. Returns the byte offset of the table's first row.
    """
    from astropy.io import fits

    first0, first1, last1 = ffi_hdrs
    centre_ra, centre_dec = wcsItem.all_pix2world(x0 + size / 2, y0 + size / 2, 0)

    prim = fits.PrimaryHDU()
    skip = {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'NEXTEND', 'EXTNAME', 'EXTVER', 'CHECKSUM', 'DATASUM'}
    for card in first0.cards:
        if card.keyword not in skip and card.keyword:
            prim.header.append(card)
    prim.header['EXTEND'] = True
    prim.header['ORIGIN'] = 'STScI/MAST'     # as astrocut writes: what lightkurve recognises a TESScut-style TPF by
    prim.header['CREATOR'] = 'TESSELLATE direct cutter'
    prim.header['TSTART'] = first1['TSTART']
    prim.header['TSTOP'] = last1['TSTOP']
    prim.header['DATE-OBS'] = first1.get('DATE-OBS', '')
    prim.header['DATE-END'] = last1.get('DATE-END', '')
    prim.header['OBJECT'] = f'sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
    prim.header['SECTOR'] = sector
    prim.header['CAMERA'] = cam
    prim.header['CCD'] = ccd
    prim.header['RA_OBJ'] = (float(centre_ra), '[deg] right ascension of cut centre')
    prim.header['DEC_OBJ'] = (float(centre_dec), '[deg] declination of cut centre')

    table = fits.BinTableHDU.from_columns(_tpf_columns(size, name_len), nrows=0, name='PIXELS')
    th = table.header
    th['NAXIS2'] = nrows
    for key in ['TIMEREF', 'TIMESYS', 'BJDREFI', 'BJDREFF', 'TIMEUNIT', 'TIMEPIXR', 'TIERRELA', 'INT_TIME',
                'FRAMETIM', 'NUM_FRM', 'TIMEDEL', 'BACKAPP', 'DEADAPP', 'VIGNAPP', 'GAINA', 'GAINB', 'GAINC',
                'GAIND', 'READNOIA', 'READNOIB', 'READNOIC', 'READNOID', 'NREADOUT']:
        if key in first1:
            th[key] = first1[key]
    th['TSTART'] = first1['TSTART']
    th['TSTOP'] = last1['TSTOP']
    th['TELAPSE'] = last1['TSTOP'] - first1['TSTART']
    th['DATE-OBS'] = first1.get('DATE-OBS', '')
    th['DATE-END'] = last1.get('DATE-END', '')
    # -- No 1CRV4P/1CRV5P (tpf.column/row) or column WCS keys, as astrocut: tpf.column must stay 0, since reduce
    # -- passes the cut's corner as col_offset itself (with column 551 too, S35 Cam3 Ccd3 Cut3 reduced with stripes)

    row_bytes = _tpf_row_dtype(size, name_len).itemsize
    if th['NAXIS1'] != row_bytes:
        raise RuntimeError(f'TPF row layout mismatch: header {th["NAXIS1"]} bytes, dtype {row_bytes}')

    # -- WCS as astrocut: a linear fit to one FFI's WCS across the cut (tpf.wcs comes from here) -- #
    fit, msep, sig = _cut_linear_wcs(wcsItem, x0, y0, size)
    # -- (DataProcessor.stamp_cut_wcs later replaces it with TessFixer's, in the same fixed-size header) -- #
    fit, msep, sig = _cut_linear_wcs(wcsItem, x0, y0, size)
    aperture = _aperture_header(size, fit.to_header(), {
        'WCS_FFI': (wcs_ffi, 'FFI of cut WCS'),
        'WCS_MSEP': (msep, '[deg] max cut WCS fit offset'),
        'WCS_SIG': (sig, '[deg] rms cut WCS fit offset'),
        'WCS_SRC': ('FFI linear fit', 'origin of this WCS')})
    th['WCS_FFI'] = (wcs_ffi, 'FFI of cut WCS')

    head = prim.header.tostring().encode() + th.tostring().encode()
    data_len = nrows * row_bytes
    ap_data = np.ones((size, size), '>i4').tobytes()
    with open(path, 'wb') as f:
        f.write(head)
        f.seek(len(head) + data_len)
        f.write(b'\0' * ((-data_len) % 2880))
        f.write(_fixed_header_bytes(aperture, APERTURE_HEADER_BLOCKS))
        f.write(ap_data)
        f.write(b'\0' * ((-len(ap_data)) % 2880))
    return len(head)


def _direct_cut_chunk(files, frames, rows, parts, targets, corners, size, name_len):
    """
    Read a run of FFIs once each and write their cut images as rows of the target TPFs (os.pwrite at fixed
    offsets, so workers never touch the same bytes). targets[part] = [(path, data_start), ...] matching corners.
    Returns each frame's block medians and header times/quality, for the frame-stats file.
    """
    from astropy.io import fits

    row_dtype = _tpf_row_dtype(size, name_len)
    rec = np.zeros(1, dtype=row_dtype)
    fds = {}
    block = 2048 // FRAME_STAT_BLOCKS
    out = dict(frames=np.asarray(frames), medians=np.full((len(files), FRAME_STAT_BLOCKS**2), np.nan, np.float32),
               dquality=np.zeros(len(files), np.int32), tstart=np.zeros(len(files)), tstop=np.zeros(len(files)))
    try:
        for p, target in enumerate(targets):
            for k, (path, _) in enumerate(target):
                fds[(p, k)] = os.open(path, os.O_WRONLY)
        for i, (file, frame, row, p) in enumerate(zip(files, frames, rows, parts)):
            try:
                with fits.open(file, memmap=False) as h:
                    hdr = h[1].header
                    cadence = h[0].header.get('FFIINDEX', frame)
                    flux = h[1].data.astype(np.float32)
                    err = h[2].data.astype(np.float32)
            except Exception as e:
                raise RuntimeError(f'Could not read FFI {file}: {e}') from e
            rec['TIME'] = 0.5 * (hdr['TSTART'] + hdr['TSTOP'])
            rec['TIMECORR'] = hdr.get('BARYCORR', 0.0)
            rec['CADENCENO'] = cadence
            rec['QUALITY'] = hdr.get('DQUALITY', 0)
            rec['FFI_FILE'] = os.path.basename(file)
            for k, (x0, y0) in enumerate(corners):
                rec['FLUX'][0] = flux[y0:y0+size, x0:x0+size]
                rec['FLUX_ERR'][0] = err[y0:y0+size, x0:x0+size]
                os.pwrite(fds[(p, k)], rec.tobytes(), targets[p][k][1] + row * row_dtype.itemsize)
            # -- same blocks and order as TessFixer.reference_candidates -- #
            out['medians'][i] = [np.nanmedian(flux[block*a:block*(a+1), 44+block*b:44+block*(b+1)])
                                 for a in range(FRAME_STAT_BLOCKS) for b in range(FRAME_STAT_BLOCKS)]
            out['dquality'][i] = hdr.get('DQUALITY', 0)
            out['tstart'][i], out['tstop'][i] = hdr['TSTART'], hdr['TSTOP']
    finally:
        for fd in fds.values():
            os.close(fd)
    return out

class DataProcessor():

    def __init__(self,sector,verbose=1,data_path=None) -> None:

        self.sector = sector
        self.verbose = verbose

        if data_path[-1] == '/':
            data_path = data_path[:-1]
        self.data_path = data_path

        self._make_path(False)

    def _make_path(self,delete):
        """
        Creates a folder for the path. 
        """

        if self.data_path is None:
            _Save_space('temporary',delete=delete)
            self.path = './temporary'
        else:
            _Save_space(f'{self.data_path}/Sector{self.sector}')
            self.path = f'{self.data_path}/Sector{self.sector}'

    def download(self,cam,ccd,number='all',time=None,single=None):
        """
        Function for downloading FFIs from MAST archive.

        ------
        Inputs
        ------
        cam : int
            specific camera, default None
        ccd : int
            desired ccd, default None

        -------
        Options:
        -------
        number : int
            if not None, downloads this many
        time : float (MJD)
            if not None, downloads only FFIs within a day of this time
        
        """

        from .downloader import Download_cam_ccd_FFIs
        
        if self.verbose > 0:
            print(_Print_buff(50,f'Downloading Sector {self.sector} Cam {cam} CCD {ccd}'))
        Download_cam_ccd_FFIs(self.path,self.sector,cam,ccd,time,None,None,number=number,single=single) 
    
    def find_cuts(self,cam,ccd,n,overlap=10,plot=True,proj=True,coord=None,verbose=1):
        """
        Function for finding cuts.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts will be made

        -------
        Options:
        -------
        plot : bool
            if True, plot cuts 
        proj : bool
            if True, plot cuts with WCS grid
        coord : (ra,dec)
            if not None, plots coord 
        
        """

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from astropy.wcs import WCS
        from astropy.io import fits
        from glob import glob

        newpath = f'{self.path}/Cam{cam}/Ccd{ccd}'

        # wcsItem = WCS(fits.open(f'{newpath}/wcs/ref/corrected.fits')[1].header)

        wcsItem = find_good_wcs(self.sector,self.data_path,cam,ccd,verbose=verbose)

        # wcsItem = _get_wcs(f'{newpath}/image_files',f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits') 
        # if not os.path.exists(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits'):
        #     wcs_save = wcsItem.to_fits()
        #     wcs_save.writeto(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits')

        if wcsItem is None:
            if verbose > 0:
                print('WCS Extraction Failed')
            return

        cutCorners, cutCentrePx, cutCentreCoords, cutSize = _cut_properties(wcsItem,n,overlap)

        if plot:
            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=wcsItem)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
            else:
                ax = plt.subplot()

            if coord is not None:
                coordPx = wcsItem.all_world2pix(coord[0],coord[1],0)
                ax.scatter(coordPx[0],coordPx[1],s=10)
                ax.text(x=coordPx[0],y=coordPx[1]+10,s=f'{coordPx[0]:.1f},{coordPx[1]:.1f}')
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,0), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            ax.set_title(f'Camera {cam} CCD {ccd}')
            ax.set_xlim(0,2136)
            ax.set_ylim(0,2078)
            ax.grid()

            ax.add_patch(rectangleTotal)
                
            # -- Adds cuts -- #
            colours = iter(plt.cm.rainbow(np.linspace(0, 1, n**2)))

            for corner in cutCorners:
                c = next(colours)
                rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                              facecolor='none',alpha=1)
                ax.add_patch(rectangle)
                
        return cutCorners, cutCentrePx, cutCentreCoords, cutSize
    
    def _make_part_cube(self,cam,ccd,input_files):

        from astrocut import CubeFactory
        from astropy.time import Time
        import datetime

        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        
        dates = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            dates.append(imagetime)

        sortedDates = np.array(sorted(dates))
        differences = np.diff(sortedDates)
        idx = np.where(differences == np.nanmax(differences))[0][0]+1

        cube1_files = []
        cube2_files = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            if imagetime in sortedDates[:idx]:
                cube1_files.append(f)
            elif imagetime in sortedDates[idx:]:
                cube2_files.append(f)  
        cubes = [cube1_files,cube2_files]

        for i in range(2):
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            if not os.path.exists(f'{broad_path}/Part{i+1}'):
                os.mkdir(f'{broad_path}/Part{i+1}')
            cube_path = f'{broad_path}/Part{i+1}/{cube_name}'

            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd} Part {i+1}'))

            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(cubes[i],cube_file=cube_path,verbose=self.verbose>1,max_memory=500)
            print('\n')


    def make_cube(self,cam,ccd,part=False):
        """
        Make cube for this cam,ccd.
        
        ------
        Inputs
        ------
        cam : int
            desired camera 
        ccd : int
            desired ccd

        -------
        Options
        -------
        delete_files : bool  
            deletes all FITS files once cube is made
        cubing_message : str
            custom printout message for self.verbose > 0

        -------
        Creates
        -------
        Data cube fits file in path.

        """

        from astrocut import CubeFactory
        from glob import glob


        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        file_path = f'{broad_path}/image_files'

        # if os.path.exists(cube_path):
        #     print(f'Cam {cam} CCD {ccd} cube already exists!')
        #     return

        input_files = glob(f'{file_path}/*ffic.fits')  # list of fits files in path
        if len(input_files) < 1:
            print('No files to cube!')
            return  
        
        deleted = _Remove_emptys(input_files)  # remove empty downloaded fits files
        if self.verbose > 1:
            print(f'Deleted {deleted} corrupted file/s.')
                    
        input_files = glob(f'{file_path}/*ffic.fits')  # regather list of good fits files
        if len(input_files) < 1:
            print('No files to cube!')
            return    

        if self.verbose > 1:
            print(f'Number of files to cube = {len(input_files)}')
            size = len(input_files) * 0.0355
            print(f'Estimated cube size = {size:.2f} GB')

        if part:
            self._make_part_cube(cam,ccd,input_files)
        else:
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{broad_path}/{cube_name}'

            # -- Allows for a custom cubing message (kinda dumb) -- #
            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd}'))
            
            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(input_files,cube_file=cube_path,verbose=self.verbose>1,max_memory=500)

    def _make_part_cuts(self,cam,ccd,n,cut,file_path,cutCentreCoords, cutSize):

        from astrocut import CutoutFactory

        for i in range(2):

            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/Part{i+1}/{cube_name}'
        
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} Cut {cut} (of {n**2}) Part {i+1}')
        
            my_cutter = CutoutFactory() # astrocut class
            coords = cutCentreCoords[cut-1]

            _Save_space(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}')
                        
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                f"{coords[0]} {coords[1]}", 
                                                (cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} Cut {cut} Part {i+1} complete.')
                print('\n')

            with open(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}/cut.txt', 'w') as file:
                file.write('Cut!')
    
    def make_cuts(self,cam,ccd,n,cut,part=False):
        """
        Make cut(s) for this CCD.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd

        ------
        Creates
        ------
        Save files for cut(s) in path.

        """

        from astrocut import CutoutFactory

        try:
            _, cutCentrePx, _, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False)
        except:
            print('Something wrong with finding cuts!')
            return
        
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        if not os.path.exists(file_path):
            print('No data to cut!')
            return
        
        if part:
            self._make_part_cuts(cam,ccd,n,cut,file_path,cutCentrePx,cutSize)
        else:   
            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/{cube_name}'
            
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            # if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
            #     print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
            # else:
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
            
            my_cutter = CutoutFactory() # astrocut class
            px = cutCentrePx[cut-1]

            _Save_space(f'{file_path}/Cut{cut}of{n**2}')
                            
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                xy_pos=(px[0],px[1]), 
                                                cutout_size=(cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
                print('\n')

    def make_direct_cuts(self,cam,ccd,n,cuts=None,part=False,wcs_ffi=None,n_workers=None,chunk=32):
        """
        Make cut TPFs for this CCD straight from its FFIs, replacing make_cube + make_cuts.

        Each FFI is read once and its cuts written as one row of each cut's TPF, so memory stays at a few FFIs
        per worker and no cube is ever written. Also saves the per-frame block medians and quality flags
        (sector*_frame_stats.npz) that TessFixer ranks reference frames on, so fix_wcs doesn't need a cube.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts on the CCD

        -------
        Options
        -------
        cuts : list of int
            cut numbers to write (default all n**2). Cuts whose cut.txt already exists are skipped.
        part : bool
            if True, split at the largest time gap into Part1/Part2 TPFs (as _make_part_cube)
        wcs_ffi : str
            file name of the FFI whose WCS the cuts' WCS is fitted to (default: the first FFI with a good WCS).
            Set it to an astrocut cut's WCS_FFI to reproduce that cut's WCS exactly.
        n_workers : int
            parallel workers (default: the CPUs this job may use)
        chunk : int
            FFIs per parallel task

        -------
        Creates
        -------
        [Part{i}/]Cut{cut}of{n**2}/sector*_cut*.fits and cut.txt for each cut; sector*_frame_stats.npz.

        """

        import shutil
        from glob import glob
        import warnings
        from astropy.io import fits
        from astropy.wcs import WCS, FITSFixedWarning
        from joblib import Parallel, delayed
        from tqdm import tqdm
        from .asteroid_prediction import _available_cpus

        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        file_path = f'{broad_path}/image_files'

        input_files = glob(f'{file_path}/*ffic.fits')
        if len(input_files) < 1:
            print('No files to cut!')
            return
        deleted = _Remove_emptys(input_files)  # remove empty downloaded fits files
        if self.verbose > 1:
            print(f'Deleted {deleted} corrupted file/s.')

        # -- sorted by name = by time: the frame order TessFixer's reference index refers to -- #
        files = sorted(glob(f'{file_path}/*ffic.fits'))
        if len(files) < 1:
            print('No files to cut!')
            return
        times = np.array([_ffi_time(f) for f in files])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f'FFI file names in {file_path} are not in strictly increasing time order')
        n_frames = len(files)

        # -- Parts: split after the largest gap -- #
        if part:
            split = int(np.argmax(np.diff(times))) + 1
            part_of = (np.arange(n_frames) >= split).astype(int)
            part_dirs = [f'{broad_path}/Part1', f'{broad_path}/Part2']
            part_starts = [0, split]
        else:
            part_of = np.zeros(n_frames, int)
            part_dirs = [broad_path]
            part_starts = [0]
        row_of = np.arange(n_frames) - np.array(part_starts)[part_of]
        part_rows = np.bincount(part_of, minlength=len(part_dirs))

        # -- Cut geometry, as make_cuts -- #
        warnings.simplefilter('ignore', FITSFixedWarning)     # astropy's 'datfix' note on every FFI header
        cutCorners, _, _, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)

        # -- The raw FFI WCS the cut WCS is fitted to, as astrocut: never the TessFixer one, so reruns agree -- #
        wcsItem = None
        for f in ([f'{file_path}/{wcs_ffi}'] if wcs_ffi else files):
            with fits.open(f) as h:
                w = WCS(h[1].header)
            if _wcs_ok(w):
                wcs_ffi, wcsItem = os.path.basename(f), w
                break
            if wcs_ffi:
                raise ValueError(f'{wcs_ffi} has no usable WCS')
        if wcsItem is None:
            raise ValueError(f'No FFI in {file_path} has a usable WCS')
        size = int(2 * cutSize)
        cuts = list(range(1, n**2 + 1)) if cuts is None else [int(c) for c in cuts]
        todo = [c for c in cuts
                if not all(os.path.exists(f'{d}/Cut{c}of{n**2}/cut.txt') for d in part_dirs)]
        stats_path = _frame_stats_path(broad_path, self.sector, cam, ccd)
        if not todo and os.path.exists(stats_path):
            print(f'Cam {cam} CCD {ccd} cuts {cuts} already made!')
            return
        corners = [tuple(int(v) for v in cutCorners[c-1]) for c in todo]

        name_len = max(len(os.path.basename(f)) for f in files)
        row_bytes = _tpf_row_dtype(size, name_len).itemsize
        need = len(todo) * n_frames * row_bytes
        free = shutil.disk_usage(broad_path).free
        if self.verbose > 0:
            print(_Print_buff(50,f'Direct Cutting Sector {self.sector} Cam {cam} CCD {ccd}'))
            print(f'WCS from {wcs_ffi}')
            print(f'{n_frames} FFIs -> {len(todo)} cuts of {size}x{size} px'
                  + (f' in 2 parts ({part_rows[0]} + {part_rows[1]} frames)' if part else '')
                  + f', {need/1e9:.1f} GB to write ({free/1e9:.0f} GB free)')
        if need > free:
            raise OSError(f'Not enough disk for the cuts: need {need/1e9:.1f} GB, {free/1e9:.1f} GB free')

        # -- Headers + empty tables, for the workers to fill row by row -- #
        with fits.open(files[0]) as h:
            first0 = h[0].header.copy()
        ffi_hdrs = []
        for p in range(len(part_dirs)):
            ffi_hdrs.append((first0, fits.getheader(files[part_starts[p]], 1),
                             fits.getheader(files[part_starts[p] + part_rows[p] - 1], 1)))
        targets = []
        for p, d in enumerate(part_dirs):
            target = []
            for c, (x0, y0) in zip(todo, corners):
                folder = f'{d}/Cut{c}of{n**2}'
                _Save_space(folder)
                path = f'{folder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{c}_of{n**2}.fits'
                start = _create_direct_tpf(path, ffi_hdrs[p], wcsItem, wcs_ffi, x0, y0, size, int(part_rows[p]), name_len,
                                           self.sector, cam, ccd, c, n)
                target.append((path, start))
            targets.append(target)

        # -- One pass over the FFIs -- #
        n_workers = _available_cpus() if n_workers is None else n_workers
        tasks = [np.arange(s, min(s + chunk, n_frames)) for s in range(0, n_frames, chunk)]
        medians = np.full((n_frames, FRAME_STAT_BLOCKS**2), np.nan, np.float32)
        dquality = np.zeros(n_frames, np.int32)
        tstart = np.zeros(n_frames)
        tstop = np.zeros(n_frames)
        done = np.zeros(n_frames, bool)
        ts = t()
        results = Parallel(n_jobs=n_workers, return_as='generator')(
            delayed(_direct_cut_chunk)([files[i] for i in idx], idx, row_of[idx], part_of[idx],
                                       targets, corners, size, name_len) for idx in tasks)
        with tqdm(total=n_frames, desc=f'Cam {cam} CCD {ccd} FFIs', unit='ffi', disable=self.verbose < 1) as bar:
            for res in results:
                f = res['frames']
                medians[f], dquality[f], tstart[f], tstop[f] = res['medians'], res['dquality'], res['tstart'], res['tstop']
                done[f] = True
                bar.update(len(f))
        if not done.all():
            raise RuntimeError(f'{(~done).sum()} FFIs were not cut')

        np.savez(stats_path, medians=medians, dquality=dquality, tstart=tstart, tstop=tstop,
                 files=np.array([os.path.basename(f) for f in files]), blocks=FRAME_STAT_BLOCKS)
        for d in part_dirs:
            for c in todo:
                with open(f'{d}/Cut{c}of{n**2}/cut.txt', 'w') as file:
                    file.write('Cut!')

        if self.verbose > 0:
            print(f'Cam {cam} CCD {ccd} direct cuts complete ({(t()-ts)/60:.1f} min, {n_workers} workers).')
            print('\n')

    def stamp_cut_wcs(self,cam,ccd,n,cut,part=False):
        """
        Write TessFixer's corrected WCS (as CutWCS gives it for this cut) into a direct cut's TPF, replacing the
        linear FFI fit, so tessreduce and lightkurve (tpf.wcs) place sources with the corrected WCS.
        Needs fix_wcs done (wcs/ref/corrected.fits and ccd_sourcefits.csv). In place: only the header changes.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts on the CCD
        cut : int
            cut number

        -------
        Options
        -------
        part : bool or int
            True stamps both Part1 and Part2 TPFs, 1 or 2 just that part
        """

        from astropy.io import fits
        from .localisation import CutWCS

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        if not os.path.exists(f'{file_path}/wcs/ref/corrected.fits'):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        cut_wcs = CutWCS(self.data_path,self.sector,cam,ccd,cut,n)
        name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
        if part is True:
            folders = [f'{file_path}/Part{i}/Cut{cut}of{n**2}' for i in (1, 2)]
        elif part:
            folders = [f'{file_path}/Part{int(part)}/Cut{cut}of{n**2}']
        else:
            folders = [f'{file_path}/Cut{cut}of{n**2}']
        for folder in folders:
            path = f'{folder}/{name}'
            hdr = fits.getheader(path, 2)
            size = hdr['NAXIS1']
            extra = {k: (hdr[k], hdr.comments[k]) for k in ('WCS_FFI',) if k in hdr}
            extra['WCS_SRC'] = ('TessFixer', 'wcs/ref/corrected.fits + cut offset')
            extra['WCS_DX'] = (float(cut_wcs.dx), '[pix] CutWCS source offset x')
            extra['WCS_DY'] = (float(cut_wcs.dy), '[pix] CutWCS source offset y')
            _stamp_tpf_wcs(path, _cutwcs_header(cut_wcs, size), extra)
            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} cut {cut}: corrected WCS written to {name} '
                      f'(offset {cut_wcs.dx:.2f}, {cut_wcs.dy:.2f} px)')

    def predict_asteroids(self,cam,ccd,n,cut,part=False,plot_trails=False):
        """
        Predicts every catalogued minor planet (from MPCORB) crossing this
        cut's footprint at any point during its observing window, using
        ASSIST-perturbed orbit propagation (see asteroid_prediction.py).
        Runs ahead of reduce() -- only needs the cut's sky footprint (from
        find_cuts) and frame times, both already available once make_cuts()
        has produced the cut's TPF (Times.npy isn't written until reduce()).

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts have been made
        cut : int
            specific cut

        ------
        Creates
        ------
        Per-cut asteroid ephemeris CSV and trail plot in the cut's folder.

        """

        from astropy.io import fits
        from astropy.wcs import WCS
        from .asteroid_prediction import predict_asteroids_for_footprint, plot_asteroid_trails

        try:
            cutCorners, _, cutCentreCoords, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)
        except:
            print('Something wrong with finding cuts!')
            return

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'

        wcs_path = f'{file_path}/wcs/ref/corrected.fits'
        if not os.path.exists(wcs_path):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        with fits.open(wcs_path) as f:
            wcsItem = WCS(f[1].header)

        ra_center, dec_center = cutCentreCoords[cut-1]
        cut_corner = cutCorners[cut-1]
        # circumscribe the square cut (half its diagonal), not just inscribe it (half its
        # side), or objects crossing near the corners get silently missed
        radius_deg = cutSize * np.sqrt(2) * 21.0 / 3600.0

        for i in range(2 if part else 1):
            part_label = f' Part {i+1}' if part else ''
            cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
            base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
            times_path = f'{cutFolder}/{base}_Times.npy'

            # the frame times saved by the reduction -- no cut data is loaded
            if os.path.exists(times_path):
                mjd = np.load(times_path)
            else:
                print(f'No saved frame times found to predict asteroids for '
                      f'(Cam {cam} Ccd {ccd} Cut {cut}{part_label})!')
                continue

            if self.verbose > 0:
                print(f'Predicting Asteroids for Cam {cam} CCD {ccd} Cut {cut} (of {n**2}){part_label}')

            try:
                # allow_download=False: this runs as a SLURM job on a compute node with no
                # internet access, so missing/uncovered kernel data must fail fast here rather
                # than have a worker process try (and fail) to reach MAST mid-integration
                ephemeris = predict_asteroids_for_footprint(
                    ra_center, dec_center, radius_deg,
                    mjd.min(), mjd.max(), mjd, wcsItem,
                    data_dir=f'{self.data_path}/mpc', allow_download=False)
            except RuntimeError as e:
                print(f'Cannot predict asteroids for Cam {cam} Ccd {ccd} Cut {cut}{part_label}: {e}')
                continue

            self._write_cut_asteroids(ephemeris, cutFolder, base, cut_corner, cutSize, plot_trails)

            if self.verbose > 0:
                n_tracks = ephemeris['designation'].nunique() if len(ephemeris) else 0
                print(f'Cam {cam} CCD {ccd} Cut {cut}{part_label} asteroid prediction complete '
                      f'({n_tracks} tracks found).')
                print('\n')

    def _write_cut_asteroids(self, ephemeris, cutFolder, base, cut_corner, cutSize, plot_trails=False):
        """Write one cut's asteroid ephemeris, asteroids.txt marker and, if plot_trails, the trail plot
        (8 s per cut with ~1000 tracks, so off by default) -- shared by the per-cut and the
        CCD-level prediction so both produce identical files."""
        from .asteroid_prediction import plot_asteroid_trails

        # store x,y local to the cut (matching the cut TPF's own pixel indexing),
        # not full-CCD, so results line up directly with the cut's reduced data
        if len(ephemeris):
            ephemeris['x'] -= cut_corner[0]
            ephemeris['y'] -= cut_corner[1]
            in_fov = ephemeris['x'].between(0,2*cutSize) & ephemeris['y'].between(0,2*cutSize)
        else:
            in_fov = ephemeris.index

        # every asteroid output except the asteroids.txt marker (checked directly in the
        # cut folder, matching cut.txt/reduced.txt's convention) lives in its own
        # subdirectory rather than cluttering the cut folder alongside everything else
        _Save_space(f'{cutFolder}/asteroids')
        save_table(ephemeris,f'{cutFolder}/asteroids/{base}_Asteroids.csv')
        if plot_trails:
            plot_asteroid_trails(ephemeris[in_fov],f'{cutFolder}/asteroids/{base}_AsteroidTrails.png',footprint_size=2*cutSize)

        with open(f'{cutFolder}/asteroids.txt', 'w') as file:
            file.write('Predicted!')

    def predict_asteroids_ccd(self,cam,ccd,n,cuts=None,part=False,plot_trails=False):
        """
        Asteroid prediction for a whole CCD at once, written out per cut in exactly the per-cut
        format (predict_asteroids). One catalogue pass and one ASSIST integration per object over
        every frame it spends on the CCD, instead of once per cut it crosses: on Sector 44 Cam1
        Ccd2 6.0 min against ~93 min for its 64 cut jobs (15.6x), identical positions (<= 0.008
        arcsec), and more complete -- it also catches fast near-Earth objects that cross a single
        cut between the coarse samples (2021 UQ: 21.8 deg/day, V = 16.8).

        The footprint is a circle enclosing every cut's own circle, over the union of the cuts'
        frame times. Each cut then gets the rows inside its pixel square plus CUT_MARGIN_PX (rows
        beyond it can't be measured in the cut, and were 38% of what the cuts' circles kept) at
        its own frame times, with 'frame' re-indexed to the cut's own frame order and x,y local
        to the cut. The split runs in the write workers: done serially it took 315 s of a dense
        CCD's run (Sector 44 Cam 4 Ccd 3, 66 M rows), 81% of it angular distances.

        ------
        Inputs
        ------
        cam, ccd : int
        n : int
            n**2 cuts have been made
        cuts : list of int, optional
            cuts to write (default all n**2); the prediction covers the whole CCD regardless
        part : bool
        """
        import pandas as pd
        from astropy.io import fits
        from astropy.wcs import WCS
        from .asteroid_prediction import predict_asteroids_for_footprint

        cuts = list(range(1, n**2 + 1)) if cuts is None else list(cuts)
        try:
            cutCorners, _, cutCentreCoords, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)
        except:
            print('Something wrong with finding cuts!')
            return

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        wcs_path = f'{file_path}/wcs/ref/corrected.fits'
        if not os.path.exists(wcs_path):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        with fits.open(wcs_path) as f:
            wcsItem = WCS(f[1].header)

        # each cut's own circle, as predict_asteroids: half the cut diagonal
        radius_deg = cutSize * np.sqrt(2) * 21.0 / 3600.0
        cen = np.radians(np.asarray(cutCentreCoords, dtype=float))
        uv = np.column_stack([np.cos(cen[:, 1]) * np.cos(cen[:, 0]), np.cos(cen[:, 1]) * np.sin(cen[:, 0]), np.sin(cen[:, 1])])
        mid = uv.mean(axis=0)
        mid /= np.linalg.norm(mid)
        ccd_ra = np.degrees(np.arctan2(mid[1], mid[0])) % 360
        ccd_dec = np.degrees(np.arcsin(mid[2]))
        ccd_radius = float(np.degrees(np.arccos(np.clip(uv @ mid, -1, 1))).max() + radius_deg)

        for i in range(2 if part else 1):
            part_label = f' Part {i+1}' if part else ''
            cut_mjds = {}
            for cut in cuts:
                cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
                base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
                times_path = f'{cutFolder}/{base}_Times.npy'
                # the frame times saved by the reduction -- no cut data is loaded
                if os.path.exists(times_path):
                    cut_mjds[cut] = np.load(times_path)
                else:
                    print(f'No saved frame times found to predict asteroids for '
                          f'(Cam {cam} Ccd {ccd} Cut {cut}{part_label})!')
            if not cut_mjds:
                continue
            mjd = np.unique(np.concatenate(list(cut_mjds.values())))

            if self.verbose > 0:
                print(f'Predicting Asteroids for Cam {cam} CCD {ccd}{part_label} ({len(cut_mjds)} cuts, '
                      f'{len(mjd)} frames, footprint radius {ccd_radius:.2f} deg)')
            try:
                # allow_download=False: SLURM compute nodes have no internet access (see predict_asteroids)
                ephemeris = predict_asteroids_for_footprint(
                    ccd_ra, ccd_dec, ccd_radius, mjd.min(), mjd.max(), mjd, wcsItem,
                    data_dir=f'{self.data_path}/mpc', allow_download=False)
            except RuntimeError as e:
                print(f'Cannot predict asteroids for Cam {cam} Ccd {ccd}{part_label}: {e}')
                continue

            # split into cuts and write them in parallel workers (_write_cuts_parallel)
            n_tracks = ephemeris['designation'].nunique() if len(ephemeris) else 0
            ccd_eph = _ccd_ephemeris_for_cuts(ephemeris, mjd, cutSize)
            del ephemeris
            jobs = []
            for cut, cmjd in cut_mjds.items():
                cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
                base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
                jobs.append((cutCorners[cut-1], cmjd, cutFolder, base, plot_trails))
            _write_cuts_parallel(self, ccd_eph, jobs)
            del ccd_eph
            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd}{part_label} asteroid prediction complete '
                      f'({n_tracks} tracks over the CCD, written to {len(cut_mjds)} cuts).')
                print('\n')

    def _reduce_part_cuts(self,cam,ccd,n,cut,filepath):

        for i in range(2):
            cutFolder = f'{filepath}/Part{i+1}/Cut{cut}of{n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            cutPath = f'{cutFolder}/{cutName}'
            fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'
            if os.path.exists(fluxName):
                if self.verbose > 0:
                    print(f'Cam {cam} Chip {ccd} Cut {cut} Part {i+1} already reduced!')
            else:
                ts = t()
                if self.verbose > 0:
                    print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2}) Part {i+1} --')

                # -- Defining so can be deleted if failed -- #
                tessreduce = 0

                # -- reduce -- #
                tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,reduce=True,corr_correction=True,
                                            calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',
                                            prf_path='/fred/oz335/_local_TESS_PRFs',backend='multiprocessing',verbose=2)
                
                if self.verbose > 0:
                    print(f'--Reduction Part {i+1} Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('\n')
                #tw = t()   # write timeStart
                
                # -- Saves information out as Numpy Arrays -- #
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy',tessreduce.lc[0])
                save_compact_array(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
                save_compact_array(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Background.npy',tessreduce.bkg)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Ref.npy',tessreduce.ref)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy',tessreduce.mask)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Shifts.npy',tessreduce.shift)

                del (tessreduce)

                with open(f'{cutFolder}/reduced.txt', 'w') as file:
                    file.write(f'Reduced with TESSreduce version {tr.__version__}.')


    def reduce(self,cam,ccd,n,cut,part=False,injection=False,injection_dir='source_injection'):
        """
        Reduces a cut on a ccd using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 part cuts

        -------
        Creates
        -------
        Fits file in path with reduced data.

        """

        import tessreduce as tr
        from .localisation import CutWCS
        
        filepath = f'{self.path}/Cam{cam}/Ccd{ccd}'
        injection_dir = injection_dir if injection else '.'

        if part:
            self._reduce_part_cuts(cam,ccd,n,cut,filepath)
        else:
            cutFolder = f'{filepath}/Cut{cut}of{n**2}'
            cutBase = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}' 

            if injection:
                cutPath = None
                save_path = f'{cutFolder}/{injection_dir}/{cutBase}'
                flux = np.load(f'{save_path}_RawFlux.npy')
                mjd = np.load(f'{cutFolder}/{cutBase}_Times.npy')
                shifts = np.load(f'{cutFolder}/{cutBase}_Shifts.npy')
                wcs = CutWCS(self.data_path,self.sector,cam,ccd,cut,n)
            else:
                cutPath = f'{cutFolder}/{cutBase}.fits'
                save_path = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
                flux = None
                mjd = None
                shifts = None
                wcs = None

            cut_corners,_,_,_ = self.find_cuts(cam,ccd,n,plot=False)

            with open(f'{self.path}/Cam{cam}/Ccd{ccd}/wcs/ref/reference.txt') as reffile:
                ref_ind = int(reffile.readlines()[0].split(': ')[-1])

            # fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'

            # if os.path.exists(fluxName):
            #     if self.verbose > 0:
            #         print(f'Cam {cam} Chip {ccd} cut {cut} already reduced!')
            # else:
            ts = t()
            if self.verbose > 0:
                print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2})--')
                


            # -- Defining so can be deleted if failed -- #
            tessreduce = 0

            # -- reduce -- #
            tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,camera=cam,ccd=ccd,
                                       flux=flux,mjd=mjd,shifts=shifts,wcs=wcs,
                                        reduce=True,corr_correction=True,
                                        calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',col_offset=int(cut_corners[cut-1][0]),#-44,
                                        prf_path='/fred/oz335/_local_TESS_PRFs',vector_path='/fred/oz335/_local_TESS_vectors',
                                        ref_ind=ref_ind,quality_bitmask='hard',shift_method='sep_core',smooth_motion=False,
                                        orbit_ref=True,create_lc=False,timing=True,backend='multiprocessing',verbose=2)

            # # -- reduce -- #
            # tessreduce = tr.tessreduce(tpf=cutPath,flux=fluxPath,time=timePath,ref=refPath,
            #                            sector=self.sector,reduce=True,corr_correction=True,
            #                             calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',col_offset=int(cut_corners[cut-1][0]),#-44,
            #                             prf_path='/fred/oz335/_local_TESS_PRFs',vector_path='/fred/oz335/_local_TESS_vectors',
            #                             ref_ind=ref_ind,quality_bitmask='hard',shift_method='sep_core',smooth_motion=False,
            #                             orbit_ref=True,create_lc=False,timing=True,backend='multiprocessing',verbose=2)
            
            if self.verbose > 0:
                print(f'--Reduction Complete (Time: {((t()-ts)/60):.2f} mins)--')
                print('\n')
            #tw = t()   # write timeStart
            
            # -- Refuse to save an all-NaN reduction (e.g. a NaN reference), so the cut isn't marked reduced -- #
            bad = [name for name in ('ref','flux','bkg') if not np.isfinite(getattr(tessreduce,name)).any()]
            if bad:
                e = f'Reduction of Cam {cam} Ccd {ccd} Cut {cut} failed: {", ".join(bad)} all NaN. Nothing saved.'
                raise RuntimeError(e)

            # -- Saves information out as Numpy Arrays -- #
            np.save(f'{save_path}_Times.npy',tessreduce.mjd)
            np.save(f'{save_path}_ReducedFlux.npy',tessreduce.flux.astype(np.float32))
            np.save(f'{save_path}_Background.npy',tessreduce.bkg)
            np.save(f'{save_path}_Ref.npy',tessreduce.ref)
            np.save(f'{save_path}_Mask.npy',tessreduce.mask)
            np.save(f'{save_path}_Shifts.npy',tessreduce.shift)
            np.save(f'{save_path}_OrbitSegments.npy',tessreduce.orbit_segments)
            np.savez(f'{save_path}_OrbitRefs.npz',
                     **{str(k): v for k, v in tessreduce.orbit_refs.items()})

            del (tessreduce)

    def asteroid_lightcurves(self,cam,ccd,n,cut,part=False,mark_done=True,n_workers=None):
        """
        Forced aperture + PSF photometry, pixel-phase detrending, star-
        contamination flagging, and shift-and-stack for every asteroid
        predict_asteroids() found crossing this cut (see
        asteroid_photometry.py) -- runs after reduce(), since it needs the
        reduced flux cube; predict_asteroids()'s own frame indexing (from
        the raw, unreduced cut TPF) is re-derived against the REDUCED
        cube's own frame list first, since reduction can drop bad-quality
        frames.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts have been made
        cut : int
            specific cut

        ------
        Creates
        ------
        The cut's photometry and tracks tables, staged for the sector's asteroid store
        (asteroid_store.merge_sector), and the cut's asteroid_lightcurves.txt marker.
        mark_done=False writes nothing into the cut folder (no marker), for test runs that must
        leave the existing products untouched. n_workers is passed to the track-offset fits
        (pool_offset_from_stacks); asteroid_lightcurves_ccd sets 1, as each cut has its own core.

        """

        import pandas as pd
        from astropy.io import fits
        from astropy.wcs import WCS
        from .asteroid_photometry import (forced_psf_photometry,
                                            match_ephemeris_to_reduced_frames, detrend_pixel_phase,
                                            local_gaia_cat_to_stars, flag_star_contamination,
                                            stack_summary, STACK_SIG_TARGET, pool_offset_from_stacks,
                                            track_offsets)
        from . import asteroid_store

        try:
            cutCorners, _, _, _ = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)
        except:
            print('Something wrong with finding cuts!')
            return

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        cut_corner = cutCorners[cut-1]

        wcs_path = f'{file_path}/wcs/ref/corrected.fits'
        if not os.path.exists(wcs_path):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        with fits.open(wcs_path) as f:
            wcsItem = WCS(f[1].header)

        for i in range(2 if part else 1):
            part_label = f' Part {i+1}' if part else ''
            cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
            base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'

            asteroids_path = f'{cutFolder}/asteroids/{base}_Asteroids.csv'
            if not table_exists(asteroids_path):
                print(f'No asteroid predictions found for Cam {cam} Ccd {ccd} Cut {cut}{part_label} '
                      '-- run predict_asteroids first!')
                continue

            flux_path = f'{cutFolder}/{base}_ReducedFlux.npy'
            times_path = f'{cutFolder}/{base}_Times.npy'
            if not (os.path.exists(flux_path) and os.path.exists(times_path)):
                print(f'No reduced data found for Cam {cam} Ccd {ccd} Cut {cut}{part_label} -- run reduce first!')
                continue

            if self.verbose > 0:
                print(f'Building Asteroid Lightcurves for Cam {cam} CCD {ccd} Cut {cut} (of {n**2}){part_label}')

            ephemeris = load_table(asteroids_path)
            if len(ephemeris) == 0:
                print(f'No asteroids predicted for Cam {cam} Ccd {ccd} Cut {cut}{part_label}.')
                # staged empty, so a rerun still replaces whatever this cut held in the store
                asteroid_store.write_empty_staging(self.data_path, self.sector, cam, ccd, cut, i + 1 if part else 0)
                if mark_done:
                    with open(f'{cutFolder}/asteroid_lightcurves.txt', 'w') as file:
                        file.write('No asteroids to process.')
                continue

            cube = np.load(flux_path)
            reduced_mjd = np.load(times_path)
            ephemeris = match_ephemeris_to_reduced_frames(ephemeris, reduced_mjd)

            gaia_cat_path = f'{cutFolder}/local_gaia_cat.csv'
            if os.path.exists(gaia_cat_path):
                gaia_cat = pd.read_csv(gaia_cat_path)
                stars = local_gaia_cat_to_stars(gaia_cat, wcsItem, cut_corner[0], cut_corner[1])
            else:
                stars = pd.DataFrame(columns=['x','y','mag'])

            # this cut's own AB zeropoint, if calibrate() has already run for it -- lets
            # pool_offset_from_stacks convert each track's fitted flux to a real magnitude
            # for a direct sanity check against the ephemeris's own predicted mag_expected.
            # None (uncalibrated) is a normal, handled case, not an error; the store fills it
            # at merge if calibrate() finishes first.
            zp, e_zp = asteroid_store.read_zeropoint(cutFolder)
            zp_ab = zp if np.isfinite(zp) else None

            # Step 1: find the predicted-vs-measured position offset, straight off the raw
            # predicted ephemeris -- measure_stack_centroid_offset works from raw cube stamps
            # and its own joint position+flux fit's significance (not a forced-photometry
            # SNR), so no photometry needs to run before the offset is known.
            offset_x, offset_y, n_offset_tracks, offset_diagnostics = pool_offset_from_stacks(
                ephemeris, cube, self.sector, cam, ccd, cut_corner[0], cut_corner[1], zp_ab=zp_ab,
                n_workers=n_workers)

            # both the aperture and the (non-centroiding, fixed-position) PSF fit assume the
            # given x,y IS the source, so a real predicted-vs-actual offset left uncorrected
            # silently loses flux on every measurement. The offset is per object (its own
            # ephemeris error): each track's own stack fit where it is significant, else the
            # cut's pooled offset, else none (track_offsets)
            per_track = track_offsets(ephemeris['designation'], offset_diagnostics, offset_x, offset_y)
            photometry_ephemeris = ephemeris.copy()
            photometry_ephemeris['x'] += photometry_ephemeris['designation'].map(per_track['offset_x']).to_numpy()
            photometry_ephemeris['y'] += photometry_ephemeris['designation'].map(per_track['offset_y']).to_numpy()

            # Step 2: forced photometry, exactly once, at the position from step 1.
            # Aperture photometry is currently disabled -- not part of the pipeline for now
            # (forced_aperture_photometry itself is untouched in asteroid_photometry.py, just
            # not called here).
            psf_df = forced_psf_photometry(cube, photometry_ephemeris, self.sector, cam, ccd,
                                             cut_corner[0], cut_corner[1])
            psf_df = detrend_pixel_phase(psf_df)

            # Step 3: contamination flags, on the one photometry result actually being saved
            psf_df = flag_star_contamination(psf_df, stars, flux_col='flux_detrended')
            stack_summary_df = stack_summary(psf_df)

            # staged for the sector's designation-sorted store (asteroid_store.merge_sector), not
            # saved in the cut folder; the stacked photometry is derived, so it is not stored
            part_index = i + 1 if part else 0
            asteroid_store.write_staging(
                self.data_path,
                asteroid_store.photometry_table(psf_df, ephemeris, self.sector, cam, ccd, cut, part_index,
                                                shape=cube.shape[1:], zp_ab=zp, e_zp_ab=e_zp),
                asteroid_store.tracks_table(ephemeris, psf_df, stack_summary_df, offset_diagnostics, offset_x, offset_y,
                                            n_offset_tracks, self.sector, cam, ccd, cut, part_index, per_track=per_track,
                                            zp_ab=zp, e_zp_ab=e_zp),
                self.sector, cam, ccd, cut, part_index)

            if mark_done:
                with open(f'{cutFolder}/asteroid_lightcurves.txt', 'w') as file:
                    file.write('Done!')

            if self.verbose > 0:
                n_tracks = ephemeris['designation'].nunique()
                n_robust = int(((~stack_summary_df['stacking_needed']) |
                                 (stack_summary_df['achieved_sig'] >= STACK_SIG_TARGET)).sum())
                print(f'Cam {cam} CCD {ccd} Cut {cut}{part_label} asteroid lightcurves complete '
                      f'({n_tracks} tracks, {n_robust} robustly detected).')
                print('\n')
    def asteroid_lightcurves_ccd(self,cam,ccd,n,cuts=None,part=False,mark_done=True,n_workers=None):
        """
        asteroid_lightcurves() for every requested cut of one CCD, in one job, several cuts at a
        time: one cut per core (n_workers, default the job's CPUs), each single-threaded with its
        offset fits serial. Run one after another, a CCD job used ~2 of its 8 cores, as each cut's
        work is now mostly single-process array operations. Densest cuts (largest prediction
        files) start first. Workers are spawned, not forked, with single-threaded BLAS set
        beforehand, so no worker inherits a BLAS thread pool pinned to one core.

        Cuts already done (asteroid_lightcurves.txt present, or with mark_done=False already staged
        in the store) are skipped, so a resubmitted job resumes where its predecessor stopped. A
        cut that fails doesn't stop the rest; the job raises at the end so the sector's merge
        (which waits on afterok) never runs on an incomplete sector. A worker killed outright
        (e.g. for memory) raises BrokenProcessPool at once.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from . import asteroid_store
        from .asteroid_photometry import _available_cpu_count

        cuts = list(range(1, n**2 + 1)) if cuts is None else list(cuts)
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        todo = []
        for cut in cuts:
            folders = [f'{file_path}/Part{i}/Cut{cut}of{n**2}' for i in (1, 2)] if part else [f'{file_path}/Cut{cut}of{n**2}']
            if mark_done and all(os.path.exists(f'{f}/asteroid_lightcurves.txt') for f in folders):
                continue
            if not mark_done and all(asteroid_store.is_staged(self.data_path, self.sector, cam, ccd, cut, p)
                                     for p in ((1, 2) if part else (0,))):
                continue
            size = sum(os.path.getsize(p) for f in folders
                       for p in [f'{f}/asteroids/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Asteroids.parquet']
                       if os.path.exists(p))
            todo.append((size, cut))
        todo = [cut for _, cut in sorted(todo, reverse=True)]
        n_workers = max(1, min(len(todo), n_workers or _available_cpu_count()))

        failed = []
        if n_workers == 1:
            results = (_lightcurves_cut_worker(self.sector, self.data_path, self.verbose, cam, ccd, n, cut, part, mark_done)
                       for cut in todo)
        else:
            for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"):
                os.environ[env] = "1"
            os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
            pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
            futures = [pool.submit(_lightcurves_cut_worker, self.sector, self.data_path, self.verbose, cam, ccd, n, cut,
                                   part, mark_done) for cut in todo]
            results = (f.result() for f in as_completed(futures))
        try:
            for cut, seconds, error in results:
                if error:
                    print(error)
                    failed.append(cut)
                if self.verbose > 0:
                    print(f'Cam {cam} CCD {ccd} Cut {cut}: {seconds:.0f} s', flush=True)
        finally:
            if n_workers > 1:
                pool.shutdown(cancel_futures=True)

        if failed:
            raise RuntimeError(f'Asteroid lightcurves failed for Cam {cam} CCD {ccd} cuts {sorted(failed)}')


def _lightcurves_cut_worker(sector, data_path, verbose, cam, ccd, n, cut, part, mark_done):
    """One cut's asteroid_lightcurves in a worker process (offset fits serial: the cut has one core).
    Returns (cut, seconds, error traceback or None)."""
    import traceback
    ts = t()
    try:
        DataProcessor(sector=sector, data_path=data_path, verbose=verbose).asteroid_lightcurves(
            cam=cam, ccd=ccd, n=n, cut=cut, part=part, mark_done=mark_done, n_workers=1)
        return cut, t() - ts, None
    except Exception:
        return cut, t() - ts, traceback.format_exc()
