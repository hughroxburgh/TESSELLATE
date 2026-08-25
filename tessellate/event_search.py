"""
Sector-wide Bazin event search and clustering for tessellate.

Workflow
--------
1. fit_cut_events()  : fit Bazin to every positive event in one cut and write a
   candidate CSV.  Run one per cut (e.g. as a slurm job) across a sector.
2. aggregate_fits()  : concatenate the per-cut CSVs into one table.
3. cluster_events()  : cluster the well-fit events in Bazin-parameter space
   (log tau_rise, log tau_fall by default) to see whether they form groups.
4. template_cluster(): return the group that contains the studied event, i.e.
   all events similar to it.
"""

import os
import glob
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-cut fitting (one slurm job per cut)
# ---------------------------------------------------------------------------

def fit_cut_events(sector, cam, ccd, cut, data_path='/fred/oz335/TESSdata', n=8,
                   units='mJy', method='psf', n_durations=3, min_duration=3,
                   max_events=10, overwrite=False, n_jobs=-1, savepath=None,
                   **fit_kwargs):
    """
    Fit Bazin to every positive event in one cut and return / save the table.

    Thin wrapper around Navigator.fit_events (which splits the events across
    n_jobs worker processes) so it can be driven from a slurm script.  savepath
    defaults to the cut folder.  overwrite=False skips the cut if its
    bazin_events.csv already exists.
    """
    from .navigator import Navigator

    if savepath is None:
        savepath = (f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/'
                    f'Cut{cut}of{n**2}/bazin_events.csv')
    if not overwrite and os.path.exists(savepath):
        print(f'Already fit (use overwrite=True): {savepath}')
        return None

    nav = Navigator(sector=sector, cam=cam, ccd=ccd, data_path=data_path, n=n)
    nav.gather_data(cut)
    nav.gather_results(cut)
    if nav.events is None or len(nav.events) == 0:
        print(f'No events for S{sector}C{cam}C{ccd}C{cut}.')
        return None

    df = nav.fit_events(cut=cut, units=units, method=method, n_durations=n_durations,
                        min_duration=min_duration, max_events=max_events, n_jobs=n_jobs,
                        savepath=savepath, **fit_kwargs)
    if df is not None:
        # Tag provenance so a sector-wide concatenation stays unambiguous
        df.insert(0, 'sector', sector)
        df.insert(1, 'cam', cam)
        df.insert(2, 'ccd', ccd)
        df.insert(3, 'cut', cut)
        df.to_csv(savepath, index=False)
    return df


# ---------------------------------------------------------------------------
# Sector-wide slurm dispatch (one job per cut)
# ---------------------------------------------------------------------------

def submit_sector_search(sector, cams=(1, 2, 3, 4), ccds=(1, 2, 3, 4), cuts=None,
                         n=8, data_path='/fred/oz335/TESSdata',
                         script_dir=None, log_dir=None,
                         units='mJy', method='psf', n_durations=5, min_duration=3,
                         max_events=10, overwrite=False,
                         time='01:00:00', cpus=8, mem=4, account='oz335',
                         submit=True):
    """
    Dispatch one slurm job per cut that fits Bazin to every positive event and
    writes <cut>/bazin_events.csv.  After the jobs finish, collect them with
    aggregate_fits('<data_path>/Sector<sector>') and cluster with cluster_events.

    cuts defaults to all n**2 cuts.  overwrite=False skips cuts whose
    bazin_events.csv already exists.  Set submit=False to only write the scripts.
    Returns the list of submitted job ids.
    """
    import sys
    import subprocess

    cuts = list(cuts) if cuts is not None else list(range(1, n ** 2 + 1))
    script_dir = script_dir or f'{data_path}/Sector{sector}/bazin_search_scripts'
    log_dir = log_dir or f'{data_path}/Sector{sector}/bazin_search_logs'
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    job_ids = []
    n_skip = 0
    for cam in cams:
        for ccd in ccds:
            for cut in cuts:
                out_csv = (f'{data_path}/Sector{sector}/Cam{cam}/Ccd{ccd}/'
                           f'Cut{cut}of{n**2}/bazin_events.csv')
                if not overwrite and os.path.exists(out_csv):
                    n_skip += 1
                    continue
                base = f'{script_dir}/S{sector}C{cam}C{ccd}C{cut}_bazin'
                with open(base + '.py', 'w') as f:
                    f.write(
                        'from tessellate.event_search import fit_cut_events\n'
                        f'fit_cut_events(sector={sector}, cam={cam}, ccd={ccd}, '
                        f'cut={cut}, data_path="{data_path}", n={n}, '
                        f'units="{units}", method="{method}", n_durations={n_durations}, '
                        f'min_duration={min_duration}, max_events={max_events}, '
                        f'overwrite={overwrite}, n_jobs={cpus})\n'
                    )
                with open(base + '.sh', 'w') as f:
                    f.write(
                        '#!/bin/bash\n'
                        f'#SBATCH --job-name=TESS_S{sector}_C{cam}_C{ccd}_C{cut}_bazin\n'
                        f'#SBATCH --output={log_dir}/%A_%x_out.txt\n'
                        f'#SBATCH --error={log_dir}/%A_%x_err.txt\n'
                        '#SBATCH --ntasks=1\n'
                        f'#SBATCH --time={time}\n'
                        f'#SBATCH --cpus-per-task={cpus}\n'
                        f'#SBATCH --mem-per-cpu={mem}G\n'
                        f'#SBATCH --account={account}\n\n'
                        'PYTHONUNBUFFERED=1\n'
                        f'{sys.executable} {base}.py\n'
                    )
                if submit:
                    r = subprocess.run(f'sbatch {base}.sh', shell=True,
                                       capture_output=True, text=True)
                    if r.returncode == 0 and r.stdout.strip():
                        job_ids.append(r.stdout.strip().split()[-1])
                    else:
                        print(f'sbatch failed S{sector}C{cam}C{ccd}C{cut}: '
                              f'{r.stderr.strip()}')

    print(f'{"Submitted" if submit else "Wrote"} '
          f'{len(cams)*len(ccds)*len(cuts) - n_skip} cut jobs '
          f'({len(job_ids)} submitted, {n_skip} skipped as already done).')
    return job_ids


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_fits(paths, n_jobs=-1):
    """
    Concatenate per-cut Bazin CSVs into one table.

    paths : a directory (searched recursively for bazin_events.csv), a glob
    pattern, or a list of file paths.
    n_jobs : read the CSVs in parallel (joblib); 1 = serial.
    """
    if isinstance(paths, str):
        if os.path.isdir(paths):
            files = glob.glob(os.path.join(paths, '**', 'bazin_events.csv'),
                              recursive=True)
        else:
            files = glob.glob(paths)
    else:
        files = list(paths)
    if not files:
        raise FileNotFoundError('No Bazin event CSVs found.')

    def _safe_read(f):
        try:
            d = pd.read_csv(f)
            return d if len(d) else None
        except pd.errors.EmptyDataError:
            return None
        except Exception as exc:
            print(f'  skipping unreadable {f}: {exc}')
            return None

    if n_jobs == 1 or len(files) < 4:
        frames = [_safe_read(f) for f in files]
    else:
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_safe_read)(f) for f in files)

    frames = [d for d in frames if d is not None]
    n_empty = len(files) - len(frames)
    if not frames:
        raise ValueError('All Bazin event CSVs were empty.')
    if n_empty:
        print(f'Aggregated {len(frames)} files ({n_empty} empty/skipped).')
    return pd.concat(frames, ignore_index=True)


def save_events(df, path):
    """
    Save an event table (e.g. a clustered table or a group) for later reload.
    Uses Parquet if the path ends in .parquet (preserves dtypes), else CSV.
    The cluster/stability columns are kept, so a saved group loads back intact.
    """
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f'Saved {len(df)} events -> {path}')


def load_events(path):
    """Load an event table written by save_events (Parquet or CSV)."""
    return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _quality_mask(df, delta_bic=-6.0, redchi2_max=3.0, min_asnr=5.0):
    """Well-fit, significant, Bazin-like events to include in the clustering."""
    keep = df.get('fit_ok', pd.Series(True, index=df.index)).fillna(False).to_numpy(dtype=bool)
    if delta_bic is not None and 'delta_bic' in df:
        keep &= (df['delta_bic'] <= delta_bic).fillna(False).to_numpy()
    if redchi2_max is not None and 'redchi2_region' in df:
        keep &= (df['redchi2_region'] <= redchi2_max).fillna(False).to_numpy()
    if min_asnr is not None and 'A_snr' in df:
        keep &= (df['A_snr'] >= min_asnr).fillna(False).to_numpy()
    return keep


def _feature_matrix(df, features, log, whiten, pca_variance,
                    delta_bic, redchi2_max, min_asnr, verbose=True):
    """
    Build the (keep_mask, whitened feature matrix) used for clustering.  Doing
    this once lets the (expensive) preprocessing be reused across HDBSCAN runs.
    """
    keep = _quality_mask(df, delta_bic, redchi2_max, min_asnr)
    for feat in features:
        keep &= np.isfinite(df[feat].to_numpy())
        if log:
            keep &= (df[feat].to_numpy() > 0)
    if keep.sum() < 2:
        return keep, None

    X = np.column_stack([
        np.log10(df.loc[keep, feat].to_numpy()) if log else df.loc[keep, feat].to_numpy()
        for feat in features
    ])
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    if whiten and Xs.shape[1] > 1:
        try:
            from sklearn.decomposition import PCA
            Xs = PCA(n_components=pca_variance, whiten=True,
                     random_state=0).fit_transform(Xs)
            if verbose:
                print(f'  PCA-whitened to {Xs.shape[1]} independent component(s) '
                      f'(>= {pca_variance:.0%} variance).')
        except Exception as exc:
            if verbose:
                print(f'  PCA whitening skipped ({exc}); using raw features.')
    return keep, Xs


def _hdbscan(Xs, min_cluster_size, min_samples=None, n_jobs=-1):
    """Density-cluster Xs (HDBSCAN, else DBSCAN).  Returns labels (-1 = noise)."""
    try:
        from sklearn.cluster import HDBSCAN
        try:
            return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           n_jobs=n_jobs).fit_predict(Xs)
        except TypeError:   # older sklearn without n_jobs
            return HDBSCAN(min_cluster_size=min_cluster_size,
                           min_samples=min_samples).fit_predict(Xs)
    except Exception:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=0.5, min_samples=min_cluster_size, n_jobs=n_jobs).fit_predict(Xs)


def cluster_events(df, features=('tau_rise', 'tau_fall', 'tau_ratio',
                                 'fwhm', 'rise_time', 'decay_time'),
                   log=True, whiten=True, pca_variance=0.99,
                   delta_bic=-6.0, redchi2_max=3.0, min_asnr=5.0,
                   min_cluster_size=15, min_samples=None, n_jobs=-1):
    """
    Cluster the well-fit events in Bazin-parameter space.

    Adds a 'cluster' column to df:
       >=0 : cluster label
        -1 : clustered as noise (passed quality, not in a dense group)
        -2 : excluded from clustering (failed the quality cuts)

    Density clustering (HDBSCAN if available, else DBSCAN) is used so groups need
    not be specified in advance and outliers fall out as noise.  Features are
    log-scaled (default) and PCA-whitened before clustering.  The default feature
    set is shape-only -- brightness features are excluded since events span a
    range of brightnesses.
    """
    df = df.copy()
    df['cluster'] = -2

    keep, Xs = _feature_matrix(df, features, log, whiten, pca_variance,
                               delta_bic, redchi2_max, min_asnr)
    if Xs is None or keep.sum() < min_cluster_size:
        print(f'Only {int(keep.sum())} events pass quality cuts; too few to cluster.')
        return df

    labels = _hdbscan(Xs, min_cluster_size, min_samples, n_jobs)
    df.loc[keep, 'cluster'] = labels
    nclus = len(set(labels) - {-1})
    print(f'Clustered {int(keep.sum())} events into {nclus} group(s) '
          f'({int(np.sum(labels == -1))} noise).')
    return df


def cluster_stability(df, objid, eventid, sector=None, cam=None, ccd=None, cut=None,
                      min_cluster_sizes=(10, 15, 20, 30, 50),
                      features=('tau_rise', 'tau_fall', 'tau_ratio',
                                'fwhm', 'rise_time', 'decay_time'),
                      log=True, whiten=True, pca_variance=0.99,
                      delta_bic=-6.0, redchi2_max=3.0, min_asnr=5.0,
                      min_samples=None, n_jobs=-1):
    """
    Membership stability of the template event's group.

    Re-clusters over a range of min_cluster_size values (reusing the same
    whitened features) and, for every clustered event, records the fraction of
    runs in which it shared the template's cluster.  Adds a 'stability' column
    (NaN for events not in the clustering set).  Events with stability ~1 are
    robustly grouped with the template; low values are marginal members.
    """
    df = df.copy()
    df['stability'] = np.nan

    keep, Xs = _feature_matrix(df, features, log, whiten, pca_variance,
                               delta_bic, redchi2_max, min_asnr)
    if Xs is None:
        print('Too few events pass quality cuts.')
        return df

    # Locate the template within the kept subset
    sel = (df['objid'] == objid) & (df['eventid'] == eventid)
    for col, val in (('sector', sector), ('cam', cam), ('ccd', ccd), ('cut', cut)):
        if val is not None and col in df:
            sel &= df[col] == val
    idx = np.where(sel.to_numpy())[0]
    if len(idx) == 0:
        print('Template event not found.')
        return df
    if not keep[idx[0]]:
        print('Template event failed the quality cuts (not clustered).')
        return df
    pos = int(keep[:idx[0]].sum())          # template's row in Xs

    co = np.zeros(int(keep.sum()))
    runs = 0
    for mcs in min_cluster_sizes:
        labels = _hdbscan(Xs, mcs, min_samples, n_jobs)
        tlbl = labels[pos]
        if tlbl < 0:                        # template was noise this run
            continue
        co += (labels == tlbl)
        runs += 1

    stability = co / runs if runs else co
    df.loc[keep, 'stability'] = stability
    print(f'Stability over {runs}/{len(min_cluster_sizes)} runs where the '
          f'template clustered; {int(np.sum(stability >= 0.5))} events with '
          f'stability >= 0.5.')
    return df


def template_cluster(df, objid, eventid, sector=None, cam=None, ccd=None, cut=None):
    """
    Return the subset of df in the same cluster as the template event -- i.e. all
    events similar to it.  Returns None if the template was not clustered.
    """
    sel = (df['objid'] == objid) & (df['eventid'] == eventid)
    for col, val in (('sector', sector), ('cam', cam), ('ccd', ccd), ('cut', cut)):
        if val is not None and col in df:
            sel &= df[col] == val
    match = df[sel]
    if len(match) == 0:
        print('Template event not found in the table.')
        return None
    lbl = int(match['cluster'].iloc[0])
    if lbl < 0:
        print(f'Template event was not assigned to a cluster (label {lbl}).')
        return None
    group = df[df['cluster'] == lbl].copy()
    print(f'Template is in cluster {lbl} with {len(group)} events.')
    return group


# ---------------------------------------------------------------------------
# Light curves for a group of events
# ---------------------------------------------------------------------------

def group_lightcurves(group, method='psf', units='mJy', events=None, n=None,
                      frame_buffer=-1, savedir=None, plot=False,
                      data_path='/fred/oz335/TESSdata', n_cuts=8, verbose=True):
    """
    Generate light curves for the events in a group / cluster.

    group : DataFrame with columns sector, cam, ccd, cut, objid, eventid
        (e.g. the output of template_cluster on an aggregated table).
    method : 'psf' (default) or 'aperture'.
    events : optional list of (objid, eventid) tuples to select specific events.
    n : if events is None, use up to n events (the most significant by delta_bic,
        else by stability, else the first n).  If both are None, all are used.
    frame_buffer : window either side of the event; -1 = whole light curve.
    savedir : if given, each light curve is written there via Navigator.event_lc
        as S{sector}C{cam}C{ccd}C{cut}O{objid}E{eventid}_{method}_{units}.csv.

    Returns dict keyed by (sector, cam, ccd, cut, objid, eventid) -> (t, flux, ferr).
    """
    from .navigator import Navigator

    need = {'sector', 'cam', 'ccd', 'cut', 'objid', 'eventid'}
    missing = need - set(group.columns)
    if missing:
        raise ValueError(f'group is missing columns: {sorted(missing)} '
                         '(use an aggregated table with provenance).')

    g = group.copy()
    if events is not None:
        want = {(int(o), int(e)) for o, e in events}
        g = g[[(int(o), int(e)) in want for o, e in zip(g['objid'], g['eventid'])]]
    elif n is not None:
        if 'delta_bic' in g:
            g = g.sort_values('delta_bic')
        elif 'stability' in g:
            g = g.sort_values('stability', ascending=False)
        g = g.head(n)

    out = {}
    for (sec, cam, ccd), sub in g.groupby(['sector', 'cam', 'ccd']):
        nav = Navigator(sector=int(sec), cam=int(cam), ccd=int(ccd),
                        data_path=data_path, n=n_cuts)
        for _, r in sub.sort_values('cut').iterrows():
            cut, objid, eventid = int(r.cut), int(r.objid), int(r.eventid)
            try:
                res = nav.event_lc(objid, eventid, cut=cut, method=method, units=units,
                                   frame_buffer=frame_buffer, savedir=savedir, plot=plot)
            except Exception as exc:
                if verbose:
                    print(f'  failed S{sec}C{cam}C{ccd}C{cut} O{objid}E{eventid}: {exc}')
                continue
            out[(int(sec), int(cam), int(ccd), cut, objid, eventid)] = res
            if verbose:
                print(f'  S{sec}C{cam}C{ccd}C{cut} O{objid}E{eventid} done', flush=True)

    print(f'Generated {len(out)} light curves ({method}, {units}).')
    return out


def plot_clusters(df, features=('tau_rise', 'tau_fall'), log=True,
                  template=None, savepath=None):
    """
    Scatter the clustered events in the two feature axes, coloured by cluster.
    template = (objid, eventid) marks the studied event.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    fx, fy = features[0], features[1]
    clustered = df[df['cluster'] >= -1]
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    noise = clustered[clustered['cluster'] == -1]
    ax.scatter(noise[fx], noise[fy], s=8, c='0.7', alpha=0.5, label='noise')
    for lbl in sorted(set(clustered['cluster']) - {-1}):
        g = clustered[clustered['cluster'] == lbl]
        ax.scatter(g[fx], g[fy], s=14, alpha=0.8, label=f'cluster {lbl}')

    if template is not None:
        tsel = (df['objid'] == template[0]) & (df['eventid'] == template[1])
        if tsel.any():
            tr = df[tsel].iloc[0]
            ax.scatter([tr[fx]], [tr[fy]], marker='*', s=260, c='k',
                       edgecolors='w', zorder=6, label='template')

    if log:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(fx); ax.set_ylabel(fy)
    ax.set_title('Bazin event clusters')
    ax.legend(fontsize=7)
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        print(f'Saved: {savepath}')
    else:
        plt.show()
