"""
Designation-sorted store for the asteroid photometry built by DataProcessor.asteroid_lightcurves.

Photometry is measured per cut but always read per object, so it is not kept in the cut folders.
Each lightcurve job stages one cut's measurements, and a merge job per sector folds them into one
file sorted by designation:

    {data_path}/asteroid_store/
        objects.parquet                     one row per designation: number, H, G, sectors, points
        photometry/sector{SS}.parquet       every measurement of the sector, sorted by (designation, mjd)
        tracks/sector{SS}.parquet           one row per (designation, cam, ccd, cut, part): stacking
                                            summary, the track's own centroid fit, the position
                                            offset applied to it and its source (own, cut, none),
                                            and the cut's pooled offset
        _staging/sector{S}/...              per-cut output waiting to be merged (deleted on merge)

Row groups of ROW_GROUP_SIZE rows carry min/max statistics on the designation, so one object's
lightcurve reads one row group per sector file, and a bulk pass over every object splits the
designation range across tasks (designation_slices) instead of scanning per cut. Rerunning a
sector rewrites only that sector's files: merge_sector replaces the rows of every cut it has
staged output for and keeps the rest.

An object crossing between cuts, CCDs or cameras is one lightcurve here: its rows from every cut
are merged in time order and tagged with cam/ccd/cut. Neighbouring cuts of a CCD overlap by 10 px,
so a crossing can be measured by both at the same frame. Every merge marks all but the measurement
farthest from its cut's edge (edge_px) as overlap_duplicate; the rows are kept, so rerunning one
cut re-decides its neighbours' duplicates, and the readers skip them by default. Flux is in counts on each cut's own scale: every row carries its
cut's AB zeropoint (zp_ab, e_zp_ab; mag = zp_ab - 2.5 log10 flux) so physical_flux() can put
pieces from different cuts on one scale. Calibration runs after the lightcurves, so zeropoints
missing at staging are filled at merge, and fill_zeropoints() fills any calibrated later.

Every value is kept only to the precision its uncertainty supports (quantize, PRECISION): a
measurement with an error column (flux, zeropoint) on a power-of-two step of error/32-error/16
(error <= error/32, adding <= 0.03% to the variance), with the error itself to 0.4%; columns
without one on fixed steps far below what they are used for -- positions (predicted position plus
the cut's fitted offset) to 1/128 px and 2^-15 deg (0.11 arcsec), time to 2^-24 d (5 ms), the
predicted magnitude to 2^-14 mag, phase angle to 2^-12 deg, distances to 1.5e-5 relative,
significances to 0.4%. Catalogue H and G are stored as given. The dropped low bits compress to
nothing, and rounding is idempotent, so merges never change stored values.

Stored at native cadence. Stacking (asteroid_photometry.stack_lightcurves) and anything else
derived is recomputed on read, so changing it never needs the photometry rerun. Columns that
are ratios or rounding of stored columns (sig, sig_detrended, xfrac, yfrac) are not stored.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 65536

# bits of the photometry 'contamination_flags' column (asteroid_photometry.flag_star_contamination)
NEAR_STAR_PROXIMITY = 1
LOCAL_FLUX_EXCESS = 2
NEAR_BRIGHT_STAR = 4
_FLAG_COLUMNS = {'near_star_proximity': NEAR_STAR_PROXIMITY, 'local_flux_excess': LOCAL_FLUX_EXCESS,
                 'near_bright_star': NEAR_BRIGHT_STAR}

# precision kept per column (quantize): ('error', column) rounds to a power of two between
# error/32 and error/16; ('mantissa', bits) to 2^-(bits+1) relative; ('step', size) to a fixed step
ERROR_STEP_FRACTION = 16
ERROR_BITS = 7                      # uncertainties and significances: 0.4%
PIXEL_STEP = 2.0 ** -7              # 1/128 px (0.16 arcsec)
DEGREE_STEP = 2.0 ** -15            # 0.11 arcsec (float32 resolves this to 512 deg)
_PIXELS = ('step', PIXEL_STEP)
PRECISION = {
    'photometry': {
        'e_flux': ('mantissa', ERROR_BITS), 'e_zp_ab': ('mantissa', ERROR_BITS),
        'flux': ('error', 'e_flux'), 'flux_detrended': ('error', 'e_flux'), 'background': ('error', 'e_flux'),
        'zp_ab': ('error', 'e_zp_ab'),
        'x': _PIXELS, 'y': _PIXELS, 'edge_px': _PIXELS, 'contaminating_star_dist_px': _PIXELS,
        'ra': ('step', DEGREE_STEP), 'dec': ('step', DEGREE_STEP),
        'mjd': ('step', 2.0 ** -24),                    # 5 ms
        'mag_expected': ('step', 2.0 ** -14),           # 6e-5 mag
        'phase_deg': ('step', 2.0 ** -12),              # 2e-4 deg
        'r_au': ('mantissa', 16), 'delta_au': ('mantissa', 16),   # 7.6e-6 relative
        'contaminating_star_mag': ('step', 2.0 ** -10),  # 0.001 mag
    },
    'tracks': {
        'e_zp_ab': ('mantissa', ERROR_BITS), 'zp_ab': ('error', 'e_zp_ab'),
        'avg_sig': ('mantissa', ERROR_BITS), 'achieved_sig': ('mantissa', ERROR_BITS),
        'centroid_sig': ('mantissa', ERROR_BITS),
        'centroid_offset_x': _PIXELS, 'centroid_offset_y': _PIXELS,
        'offset_x': _PIXELS, 'offset_y': _PIXELS, 'cut_offset_x': _PIXELS, 'cut_offset_y': _PIXELS,
        'centroid_mag': ('step', 2.0 ** -10),
    },
}

PHOTOMETRY_SCHEMA = pa.schema([
    ('designation', pa.string()),
    ('mjd', pa.float64()),
    ('flux', pa.float32()),
    ('e_flux', pa.float32()),
    ('flux_detrended', pa.float32()),
    ('background', pa.float32()),
    ('x', pa.float32()),
    ('y', pa.float32()),
    ('ra', pa.float32()),     # kept to DEGREE_STEP; TESS pixels are 21 arcsec
    ('dec', pa.float32()),
    ('r_au', pa.float32()),
    ('delta_au', pa.float32()),
    ('phase_deg', pa.float32()),
    ('mag_expected', pa.float32()),
    ('zp_ab', pa.float32()),
    ('e_zp_ab', pa.float32()),
    ('edge_px', pa.float32()),
    ('contamination_flags', pa.uint8()),
    ('overlap_duplicate', pa.bool_()),
    ('contaminating_star_dist_px', pa.float32()),
    ('contaminating_star_mag', pa.float32()),
    ('sector', pa.uint8()),
    ('cam', pa.uint8()),
    ('ccd', pa.uint8()),
    ('cut', pa.uint16()),
    ('part', pa.uint8()),
    ('frame', pa.uint16()),
])

TRACKS_SCHEMA = pa.schema([
    ('designation', pa.string()),
    ('sector', pa.uint8()),
    ('cam', pa.uint8()),
    ('ccd', pa.uint8()),
    ('cut', pa.uint16()),
    ('part', pa.uint8()),
    ('magnitude_H', pa.float32()),
    ('magnitude_G', pa.float32()),
    ('zp_ab', pa.float32()),
    ('e_zp_ab', pa.float32()),
    ('n_predicted', pa.uint32()),
    ('n_points', pa.uint32()),
    ('mjd_start', pa.float64()),
    ('mjd_end', pa.float64()),
    ('avg_sig', pa.float32()),
    ('stacking_needed', pa.bool_()),
    ('n_stack', pa.float32()),
    ('n_clean', pa.uint32()),
    ('achieved_sig', pa.float32()),
    ('centroid_offset_x', pa.float32()),
    ('centroid_offset_y', pa.float32()),
    ('centroid_sig', pa.float32()),
    ('centroid_mag', pa.float32()),
    ('centroid_n_stamps', pa.float32()),
    ('centroid_used', pa.bool_()),
    ('centroid_fit', pa.string()),
    ('offset_x', pa.float32()),
    ('offset_y', pa.float32()),
    ('offset_source', pa.string()),
    ('cut_offset_x', pa.float32()),
    ('cut_offset_y', pa.float32()),
    ('cut_offset_n_tracks', pa.uint16()),
])


def store_path(data_path):
    """{data_path}/asteroid_store, or ASTEROID_STORE_DIR if set (test runs kept apart from the
    production store; cut data is still read from data_path)."""
    return os.environ.get('ASTEROID_STORE_DIR', f'{data_path}/asteroid_store')


def _staging_dir(data_path, sector):
    return f'{store_path(data_path)}/_staging/sector{sector}'


def _staging_base(data_path, sector, cam, ccd, cut, part):
    return f'{_staging_dir(data_path, sector)}/cam{cam}_ccd{ccd}_cut{cut}_part{part}'


def _to_table(df, schema):
    """DataFrame -> Table with exactly the schema's columns and types (an empty frame gives an
    empty table, so cuts with no asteroids still merge)."""
    df = df.reindex(columns=schema.names)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def _write_atomic(table, path, **kwargs):
    """Write then rename, so a job killed mid-write never leaves a truncated file behind. Float
    columns use byte-stream-split rather than dictionary encoding: measured on Sector 29 Cam 2
    Ccd 2 Cuts 23-24, 38.7 bytes/row against 77.6 with the default dictionaries."""
    floats = [f.name for f in table.schema if pa.types.is_floating(f.type)]
    others = [f.name for f in table.schema if f.name not in floats]
    tmp = f'{path}.tmp'
    pq.write_table(table, tmp, compression='zstd', use_dictionary=others, use_byte_stream_split=floats, **kwargs)
    os.replace(tmp, path)


def _round_mantissa(values, bits):
    """Round float32 values to `bits` mantissa bits (relative precision 2^-(bits+1))."""
    a = np.asarray(values, dtype=np.float32).copy()
    ok = np.isfinite(a)
    drop = 23 - bits
    i = a[ok].view(np.int32)
    a[ok] = ((i + (1 << (drop - 1))) & ~((1 << drop) - 1)).astype(np.int32).view(np.float32)
    return a


def _round_to_step(values, step):
    """Round to a multiple of step (scalar or per value; NaN or non-positive steps leave the value)."""
    a = np.asarray(values, dtype=np.float64)
    step = np.broadcast_to(np.asarray(step, dtype=np.float64), a.shape)
    ok = np.isfinite(a) & np.isfinite(step) & (step > 0)
    out = a.copy()
    out[ok] = np.round(a[ok] / step[ok]) * step[ok]
    return out


def _error_step(error):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2.0 ** np.floor(np.log2(np.asarray(error, dtype=np.float64) / ERROR_STEP_FRACTION))


def quantize(df, kind):
    """Round each column of a photometry or tracks frame to the precision in PRECISION[kind]: error
    columns (mantissa rounding) first, so the steps derived from them are the stored values and
    rounding again changes nothing."""
    out = df.copy()
    spec = PRECISION[kind]
    for col, (how, arg) in sorted(spec.items(), key=lambda item: item[1][0] != 'mantissa'):
        if col not in out.columns:
            continue
        if how == 'mantissa':
            out[col] = _round_mantissa(out[col], arg)
        elif how == 'error':
            out[col] = _round_to_step(out[col], _error_step(out[arg]))
        else:
            out[col] = _round_to_step(out[col], arg)
    return out


def read_zeropoint(cut_folder):
    """The cut's AB zeropoint and its error from calibrate(), or NaN if it hasn't been calibrated."""
    path = f'{cut_folder}/calibration/psf_calibration_zp.csv'
    try:
        zp = pd.read_csv(path).iloc[0]
        return float(zp['zp_ab']), float(zp['e_zp_ab'])
    except Exception:
        return np.nan, np.nan


def photometry_table(psf_df, ephemeris, sector, cam, ccd, cut, part=0, shape=None, zp_ab=np.nan, e_zp_ab=np.nan):
    """One cut's forced photometry (after detrend_pixel_phase and flag_star_contamination) with the
    ephemeris geometry of each measured frame joined on, in the store's photometry schema. shape is
    the cut's (ny, nx), for each point's distance to the cut edge."""
    geometry = ephemeris[['designation', 'frame', 'ra', 'dec', 'r_helio_au', 'delta_au',
                          'phase_angle_deg', 'mag_expected']].rename(
        columns={'r_helio_au': 'r_au', 'phase_angle_deg': 'phase_deg'})
    df = psf_df.merge(geometry, on=['designation', 'frame'], how='left')
    flags = np.zeros(len(df), dtype=np.uint8)
    for col, bit in _FLAG_COLUMNS.items():
        if col in df.columns:
            flags |= np.where(df[col].fillna(False).astype(bool), bit, 0).astype(np.uint8)
    df['contamination_flags'] = flags
    if shape is not None and len(df):
        ny, nx = shape
        df['edge_px'] = np.minimum.reduce([df['x'], df['y'], nx - 1 - df['x'], ny - 1 - df['y']])
    df['zp_ab'], df['e_zp_ab'] = zp_ab, e_zp_ab
    df['sector'], df['cam'], df['ccd'], df['cut'], df['part'] = sector, cam, ccd, cut, part
    df['overlap_duplicate'] = False
    return _to_table(quantize(df, 'photometry'), PHOTOMETRY_SCHEMA)


def tracks_table(ephemeris, psf_df, stack_summary, offset_diagnostics, offset_x, offset_y, n_offset_tracks,
                 sector, cam, ccd, cut, part=0, zp_ab=np.nan, e_zp_ab=np.nan, per_track=None):
    """One row per predicted track of the cut: prediction counts, the stacking summary, this track's
    centroid-offset measurement, the offset applied to it (per_track: asteroid_photometry.track_offsets)
    and the cut's pooled offset."""
    pred = ephemeris.groupby('designation').agg(magnitude_H=('magnitude_H', 'first'),
                                                magnitude_G=('magnitude_G', 'first'),
                                                n_predicted=('frame', 'size'))
    meas = psf_df.groupby('designation').agg(n_points=('mjd', 'size'), mjd_start=('mjd', 'min'),
                                             mjd_end=('mjd', 'max'))
    stack = stack_summary.rename(columns={'n_frames': 'n_clean'})
    centroid = offset_diagnostics.set_index('designation').rename(
        columns={'offset_x': 'centroid_offset_x', 'offset_y': 'centroid_offset_y', 'sig': 'centroid_sig',
                 'mag': 'centroid_mag', 'n_stamps': 'centroid_n_stamps', 'used': 'centroid_used',
                 'fit': 'centroid_fit'})
    df = pred.join(meas).join(stack).join(centroid[[c for c in centroid.columns if c.startswith('centroid_')]])
    if per_track is not None:
        df = df.join(per_track[['offset_x', 'offset_y', 'offset_source']])
    df = df.reset_index().rename(columns={'index': 'designation'})
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + TRACKS_SCHEMA.names)))
    df['n_points'] = df['n_points'].fillna(0)
    for col in ['stacking_needed', 'centroid_used']:
        df[col] = df[col].astype('boolean')
    df['n_clean'] = df['n_clean'].fillna(0)
    df['cut_offset_x'], df['cut_offset_y'], df['cut_offset_n_tracks'] = offset_x, offset_y, n_offset_tracks
    df['zp_ab'], df['e_zp_ab'] = zp_ab, e_zp_ab
    df['sector'], df['cam'], df['ccd'], df['cut'], df['part'] = sector, cam, ccd, cut, part
    return _to_table(quantize(df, 'tracks'), TRACKS_SCHEMA)


def write_staging(data_path, photometry, tracks, sector, cam, ccd, cut, part=0):
    """Stage one cut's photometry and tracks tables for merge_sector. The tracks file is written
    last, so its presence means the cut is complete."""
    os.makedirs(_staging_dir(data_path, sector), exist_ok=True)
    base = _staging_base(data_path, sector, cam, ccd, cut, part)
    _write_atomic(photometry, f'{base}_photometry.parquet')
    _write_atomic(tracks, f'{base}_tracks.parquet')


def write_empty_staging(data_path, sector, cam, ccd, cut, part=0):
    """Stage a cut with no predicted asteroids."""
    write_staging(data_path, PHOTOMETRY_SCHEMA.empty_table(), TRACKS_SCHEMA.empty_table(), sector, cam, ccd, cut, part)


def remove_staging(data_path, sector, cam, ccd, cut, part=0):
    base = _staging_base(data_path, sector, cam, ccd, cut, part)
    for kind in ['photometry', 'tracks']:
        if os.path.exists(f'{base}_{kind}.parquet'):
            os.remove(f'{base}_{kind}.parquet')


def is_staged(data_path, sector, cam, ccd, cut, part=0):
    return os.path.exists(f'{_staging_base(data_path, sector, cam, ccd, cut, part)}_tracks.parquet')


def staged_cuts(data_path, sector):
    """Staging bases with a complete (tracks file present) cut waiting to be merged."""
    return sorted(p[:-len('_tracks.parquet')]
                  for p in glob.glob(f'{_staging_dir(data_path, sector)}/*_tracks.parquet'))


def _cut_key(table):
    # one integer per (cam, ccd, cut, part), to replace a re-staged cut's rows in a sector file
    key = pc.multiply(pc.cast(table['cam'], pa.int64()), 10)
    key = pc.multiply(pc.add(key, pc.cast(table['ccd'], pa.int64())), 100000)
    key = pc.multiply(pc.add(key, pc.cast(table['cut'], pa.int64())), 10)
    return pc.add(key, pc.cast(table['part'], pa.int64()))


def _mark_overlap_duplicates(photometry):
    """Where neighbouring cuts both measured an object at the same frame, mark every measurement but
    the one farthest from its cut's edge as overlap_duplicate."""
    t = photometry.sort_by([('designation', 'ascending'), ('mjd', 'ascending'), ('edge_px', 'descending')])
    dup = np.zeros(t.num_rows, dtype=bool)
    if t.num_rows > 1:
        des = t['designation'].combine_chunks()
        dup[1:] = (pc.equal(des[1:], des[:-1]).to_numpy(zero_copy_only=False)
                   & (np.abs(np.diff(t['mjd'].to_numpy())) < 1e-6))
    return t.set_column(t.schema.get_field_index('overlap_duplicate'), 'overlap_duplicate', pa.array(dup))


def _fill_table_zeropoints(table, data_path, sector):
    """Fill missing zp_ab/e_zp_ab from each cut's calibration file. Returns (table, n_cuts_filled)."""
    zp = table['zp_ab'].to_numpy(zero_copy_only=False)
    missing = ~np.isfinite(zp)
    if not missing.any():
        return table, 0
    key = _cut_key(table).to_numpy()
    cols = {c: table[c].to_numpy() for c in ['cam', 'ccd', 'cut', 'part']}
    zp, e_zp = zp.copy(), table['e_zp_ab'].to_numpy(zero_copy_only=False).copy()
    filled = 0
    for k in np.unique(key[missing]):
        i = np.flatnonzero(key == k)
        cam, ccd, cut, part = (int(cols[c][i[0]]) for c in ['cam', 'ccd', 'cut', 'part'])
        folder = f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}' + (f'/Part{part}' if part else '')
        found = glob.glob(f'{folder}/Cut{cut}of*')
        z, e = read_zeropoint(found[0]) if len(found) == 1 else (np.nan, np.nan)
        e = float(_round_mantissa([e], ERROR_BITS)[0])
        z = float(_round_to_step([z], _error_step([e]))[0])
        if np.isfinite(z):
            zp[i], e_zp[i] = z, e
            filled += 1
    table = table.set_column(table.schema.get_field_index('zp_ab'), 'zp_ab', pa.array(zp, pa.float32()))
    table = table.set_column(table.schema.get_field_index('e_zp_ab'), 'e_zp_ab', pa.array(e_zp, pa.float32()))
    return table, filled


def _count_points(tracks, photometry):
    # n_points/mjd range per track, not counting overlap duplicates
    keys = ['designation', 'cam', 'ccd', 'cut', 'part']
    counts = photometry.filter(pc.invert(photometry['overlap_duplicate'])).group_by(keys).aggregate([('mjd', 'count'), ('mjd', 'min'), ('mjd', 'max')])
    joined = tracks.drop_columns(['n_points', 'mjd_start', 'mjd_end']).join(counts, keys=keys)
    n_points = pc.fill_null(joined['mjd_count'], 0)
    joined = joined.append_column('n_points', pc.cast(n_points, pa.uint32()))
    joined = joined.append_column('mjd_start', joined['mjd_min']).append_column('mjd_end', joined['mjd_max'])
    return joined.select(TRACKS_SCHEMA.names).cast(TRACKS_SCHEMA)


def merge_sector(data_path, sector, rebuild=True):
    """Fold the sector's staged cuts into photometry/ and tracks/ sector files, replacing the rows of
    any cut already in them; drop overlap duplicates, fill zeropoints calibrated since staging,
    then delete the merged staging files and rebuild objects.parquet."""
    bases = staged_cuts(data_path, sector)
    if not bases:
        print(f'Sector {sector}: nothing staged to merge.')
        return
    root = store_path(data_path)
    os.makedirs(f'{root}/photometry', exist_ok=True)
    os.makedirs(f'{root}/tracks', exist_ok=True)

    # the staged cuts, from their file names (a re-staged cut can be empty, with no rows to key on)
    ids = [re.search(r'cam(\d+)_ccd(\d+)_cut(\d+)_part(\d+)$', b).groups() for b in bases]
    restaged = _cut_key(pa.table({c: pa.array([int(i[k]) for i in ids], pa.int64())
                                  for k, c in enumerate(['cam', 'ccd', 'cut', 'part'])}))
    tables = {}
    for kind, schema in [('photometry', PHOTOMETRY_SCHEMA), ('tracks', TRACKS_SCHEMA)]:
        new = pa.concat_tables([pq.read_table(f'{b}_{kind}.parquet', schema=schema) for b in bases])
        path = f'{root}/{kind}/sector{sector:02d}.parquet'
        if os.path.exists(path):
            old = pq.read_table(path, schema=schema)
            new = pa.concat_tables([old.filter(pc.invert(pc.is_in(_cut_key(old), value_set=restaged))), new])
        tables[kind], filled = _fill_table_zeropoints(new, data_path, sector)
        if kind == 'photometry':
            print(f'Sector {sector}: zeropoints filled from calibration for {filled} cuts')

    tables['photometry'] = _mark_overlap_duplicates(tables['photometry'])
    print(f'Sector {sector}: {pc.sum(tables["photometry"]["overlap_duplicate"]).as_py() or 0:,} overlap duplicates marked')
    tables['tracks'] = _count_points(tables['tracks'], tables['photometry']).sort_by(
        [('designation', 'ascending'), ('cam', 'ascending'), ('ccd', 'ascending'), ('cut', 'ascending')])
    for kind, table in tables.items():
        path = f'{root}/{kind}/sector{sector:02d}.parquet'
        _write_atomic(table, path, row_group_size=ROW_GROUP_SIZE)
        print(f'Sector {sector}: {kind} {table.num_rows:,} rows -> {path}')

    for b in bases:
        os.remove(f'{b}_photometry.parquet')
        os.remove(f'{b}_tracks.parquet')
    print(f'Sector {sector}: merged {len(bases)} staged cuts.')
    if rebuild:
        rebuild_objects(data_path)


def fill_zeropoints(data_path, sector):
    """Fill zeropoints for cuts calibrated after the sector was merged; rewrites the sector's files
    only if something was filled."""
    root = store_path(data_path)
    for kind, schema in [('photometry', PHOTOMETRY_SCHEMA), ('tracks', TRACKS_SCHEMA)]:
        path = f'{root}/{kind}/sector{sector:02d}.parquet'
        if not os.path.exists(path):
            continue
        table, filled = _fill_table_zeropoints(pq.read_table(path, schema=schema), data_path, sector)
        if filled:
            _write_atomic(table, path, row_group_size=ROW_GROUP_SIZE)
        print(f'Sector {sector}: {kind} zeropoints filled for {filled} cuts')


def physical_flux(df):
    """Add flux_ujy, e_flux_ujy and flux_detrended_ujy (microjansky, from each row's own cut
    zeropoint; NaN where the cut is uncalibrated). The zeropoint error e_zp_ab is a per-cut
    systematic and is not folded into e_flux_ujy."""
    scale = 10 ** (-0.4 * (df['zp_ab'].astype(float) - 23.9))
    out = df.copy()
    for col in ['flux', 'e_flux', 'flux_detrended']:
        if col in out.columns:
            out[f'{col}_ujy'] = out[col].astype(float) * scale
    return out


def rebuild_objects(data_path):
    """objects.parquet from every sector's tracks file: one row per designation."""
    root = store_path(data_path)
    files = sorted(glob.glob(f'{root}/tracks/sector*.parquet'))
    if not files:
        return
    t = pd.concat([pd.read_parquet(f, columns=['designation', 'sector', 'magnitude_H', 'magnitude_G',
                                                'n_points', 'mjd_start', 'mjd_end']) for f in files],
                  ignore_index=True)
    obj = t.groupby('designation').agg(magnitude_H=('magnitude_H', 'first'), magnitude_G=('magnitude_G', 'first'),
                                       n_points=('n_points', 'sum'), mjd_start=('mjd_start', 'min'),
                                       mjd_end=('mjd_end', 'max'),
                                       sectors=('sector', lambda s: ','.join(str(v) for v in sorted(set(s)))))
    obj = obj.reset_index()
    number = obj['designation'].str.extract(r'^\((\d+)\)')[0]
    obj.insert(1, 'number', pd.to_numeric(number).astype('Int64'))
    obj['n_sectors'] = obj['sectors'].str.count(',') + 1
    _write_atomic(pa.Table.from_pandas(obj, preserve_index=False), f'{root}/objects.parquet')
    print(f'objects.parquet: {len(obj):,} designations from {len(files)} sectors')


def load_objects(data_path):
    return pd.read_parquet(f'{store_path(data_path)}/objects.parquet')


def _designations(data_path, designation):
    # accept designation strings or MPC numbers
    items = [designation] if np.isscalar(designation) else list(designation)
    numbers = [int(i) for i in items if isinstance(i, (int, np.integer))]
    names = [str(i) for i in items if not isinstance(i, (int, np.integer))]
    if numbers:
        obj = load_objects(data_path)
        names += obj.loc[obj['number'].isin(numbers), 'designation'].tolist()
    return names


def _dataset(data_path, kind, sectors=None):
    root = store_path(data_path)
    if sectors is None:
        files = sorted(glob.glob(f'{root}/{kind}/sector*.parquet'))
    else:
        files = [f'{root}/{kind}/sector{s:02d}.parquet' for s in np.atleast_1d(sectors)]
        files = [f for f in files if os.path.exists(f)]
    return ds.dataset(files, format='parquet')


def _skip_duplicates(expr, kind, include_duplicates):
    return expr if kind != 'photometry' or include_duplicates else expr & ~ds.field('overlap_duplicate')


def load_lightcurve(designation, data_path, sectors=None, columns=None, kind='photometry', include_duplicates=False):
    """Every stored measurement of one or more objects (designation strings or MPC numbers), sorted
    by designation and mjd, without overlap duplicates unless include_duplicates. kind='tracks'
    returns their per-cut track rows instead."""
    names = _designations(data_path, designation)
    if not names:
        return pd.DataFrame(columns=columns or PHOTOMETRY_SCHEMA.names)
    field = ds.field('designation')
    expr = _skip_duplicates(field == names[0] if len(names) == 1 else field.isin(names), kind, include_duplicates)
    df = _dataset(data_path, kind, sectors).to_table(columns=columns, filter=expr).to_pandas()
    order = [c for c in ['designation', 'mjd', 'sector', 'cam', 'ccd', 'cut'] if c in df.columns]
    return df.sort_values(order, ignore_index=True) if order else df


def designation_slices(data_path, ntasks):
    """Split the designation range into ntasks contiguous [lo, hi] slices of roughly equal point
    count, for array jobs that each process a slice with load_designation_range."""
    obj = load_objects(data_path).sort_values('designation')
    edges = np.searchsorted(obj['n_points'].cumsum().values,
                            np.linspace(0, obj['n_points'].sum(), ntasks + 1)[1:-1])
    groups = np.split(obj['designation'].values, edges)
    return [(g[0], g[-1]) for g in groups if len(g)]


def load_designation_range(lo, hi, data_path, sectors=None, columns=None, kind='photometry', include_duplicates=False):
    """Every measurement with lo <= designation <= hi (without overlap duplicates unless
    include_duplicates); reads only the overlapping row groups."""
    field = ds.field('designation')
    expr = _skip_duplicates((field >= lo) & (field <= hi), kind, include_duplicates)
    df = _dataset(data_path, kind, sectors).to_table(columns=columns, filter=expr).to_pandas()
    order = [c for c in ['designation', 'mjd', 'sector', 'cam', 'ccd', 'cut'] if c in df.columns]
    return df.sort_values(order, ignore_index=True) if order else df
