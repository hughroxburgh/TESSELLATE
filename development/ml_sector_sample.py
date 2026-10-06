"""
After ml_classify_sector.py: add the significance and a few other columns to the per-cut predictions, write
small tables to copy down, and pick a stratified random sample of the sectors' events to sort by eye -- the
final test on sectors the model never saw. Run on the cluster.

The predictions files don't have lc_sig_max; it comes from each cut's saved features file
(tab_lc_sig_max, with tab_mjd_duration_hr, ctx_ls_power, ctx_n_comparable50).

Every event of the sectors is in the population (the classify run scored them all), so each picked event
stands for N_stratum / n_picked events of its sector and stratum (column `weight`). The strata are rules,
not model scores -- like ml_pick_sort_sample.py -- so the sample doesn't depend on what the model thinks.

Outputs in OUT_DIR:
  S{s}_ml_slim.csv.gz             (SAVE_SLIM) every event: keys, pipeline tag, probabilities, pred_class, review,
                                  the columns above (stays on the cluster; ~100 bytes per event)
  S{first}-{last}_ml_sig5.csv.gz  the same for lc_sig_max >= 5 only: copy this down
  S{first}-{last}_ml_summary.txt  the model's calls by significance and by pipeline tag: copy this down
  sort_sample/S{first}-{last}_sort_sample.csv
                                  the events to plot and sort (keys only -- nothing that hints at the answer);
                                  give it to ml_plot_sort_sample.py (SAMPLE_CSV), then manual_sort
  sort_sample/S{first}-{last}_sort_sample_info.csv
                                  the same events with stratum, weight and the model's call: copy this down

Edit the CONFIG block, then:  python ml_sector_sample.py
With SLURM = True it submits itself as one job (reading ~1000 feature files per sector takes a while);
SLURM = False runs it right here.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ---- CONFIG ----
OUT_DIR = '/fred/oz335/hroxburg/dev/ml_classifier/v5/sector_tests/predictions'   # ml_classify_sector.py's OUT_DIR
SECTORS = [53, 54]
FEATURE_VERSION = 5         # the _features_v{N} files to read
EXTRA = ['tab_lc_sig_max', 'tab_mjd_duration_hr', 'ctx_ls_power', 'ctx_n_comparable50']
SAVE_SLIM = False           # True = also write every event to S{s}_ml_slim.csv.gz (S54: 25.8M events, ~2.5 GB,
                            # a 20-40 min single-threaded gzip write). Not needed for the summary or the sample

STRATA = [                  # (name, rule, n per sector): first matching rule wins. Same rules as the last sort
    ('periodic',    'lc_sig_max >= 5 and ls_power >= 0.3', 40),
    ('sig_5_10',    'lc_sig_max >= 5 and lc_sig_max < 10', 60),
    ('sig_10_plus', 'lc_sig_max >= 10', 60),
    ('below_5',     'lc_sig_max < 5', 15),
]
SEED = 53
SAMPLE_DIR = f'{OUT_DIR}/sort_sample'

N_JOBS = 8                  # reading is I/O-bound: more workers don't help much
SLURM = True
TIME = '01:00:00'           # S54 (1023 cuts): reading 0.7 min; the job ~2 min without SAVE_SLIM
MEM_PER_CPU_GB = 2          # S54's 25.8M events peaked at 9.9 GB
ACCOUNT = 'oz335'
# ----------------

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
RENAME = {'tab_lc_sig_max': 'lc_sig_max', 'tab_mjd_duration_hr': 'duration_hr', 'ctx_ls_power': 'ls_power',
          'ctx_n_comparable50': 'n_comparable50'}
PRED_COLS = KEY_COLS + ['frame_bin', 'flux_sign', 'classification', 'p_Junk', 'p_CosmicRay', 'p_Asteroid',
                        'p_Flare', 'p_Variable', 'p_astrophysical', 'pred_class', 'entropy', 'domain_out_frac',
                        'review']


def read_cut(pred_file):
    """One cut's predictions with the EXTRA feature columns; None if its features file is missing."""
    feat_file = pred_file.replace('_ml_predictions.csv', f'_features_v{FEATURE_VERSION}.csv.gz')
    pred = pd.read_csv(pred_file, low_memory=False)
    if not len(pred):
        return None
    if not os.path.exists(feat_file):
        print(f'  no features file for {os.path.basename(pred_file)}; skipped', flush=True)
        return None
    feats = pd.read_csv(feat_file, usecols=['objid', 'eventid'] + EXTRA)
    out = pred[[c for c in PRED_COLS if c in pred]].merge(feats, on=['objid', 'eventid'], how='left')
    return out.rename(columns=RENAME)


def summary(df, sector):
    lines = [f'===== Sector {sector}: {len(df)} frame-bin-1 events', '']
    sig = pd.cut(df.lc_sig_max, [-np.inf, 5, 10, 30, np.inf], right=False,
                 labels=['< 5', '5-10', '10-30', '>= 30'])
    t = pd.crosstab(sig, df.pred_class)
    t.insert(0, 'n', sig.value_counts())
    lines += ['Model calls by lc_sig_max (counts):', t.to_string(), '']
    lines += [f'review (top p < 0.75): all {df.review.mean():.1%}, sig >= 5 {df.review[df.lc_sig_max >= 5].mean():.1%}',
              '']
    s5 = df[df.lc_sig_max >= 5]
    tag = s5.classification.astype(str)
    tag = pd.Series(np.where(tag.str.startswith('V'), 'catalogue variable', np.where(tag == '-', 'untagged', tag)),
                    name='pipeline tag', index=s5.index)
    t = pd.crosstab(tag, s5.pred_class, normalize='index').round(3)
    t.insert(0, 'n', tag.value_counts())
    lines += ['lc_sig_max >= 5: pipeline tag (rows) against the model (columns), fraction of each row:',
              t.to_string(), '']
    lines += ['lc_sig_max >= 5, events per cut (crowding): ' +
              s5.groupby(['camera', 'ccd', 'cut']).size().describe()[['min', '50%', 'max']].round(0).to_string()
              .replace('\n', ', '), '']
    return '\n'.join(lines)


def pick(df, sector, rng):
    stratum = np.full(len(df), '', dtype=object)
    for name, rule, _ in STRATA:
        hit = df.eval(rule).fillna(False).to_numpy(bool) & (stratum == '')
        stratum[hit] = name
    picked = []
    for name, _, n in STRATA:
        idx = np.flatnonzero(stratum == name)
        k = min(n, len(idx))
        take = rng.choice(idx, k, replace=False) if k else np.array([], int)
        picked.append(df.iloc[take].assign(stratum=name, n_stratum=len(idx), weight=len(idx) / max(k, 1)))
        print(f'  S{sector} {name:12s} {len(idx):9d} events, {k} picked', flush=True)
    return pd.concat(picked)


def main():
    from joblib import Parallel, delayed
    rng = np.random.default_rng(SEED)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    name = f'S{SECTORS[0]}-{SECTORS[-1]}' if len(SECTORS) > 1 else f'S{SECTORS[0]}'
    texts, sig5, samples = [], [], []
    for sector in SECTORS:
        start = time.time()
        d = f'{OUT_DIR}/S{sector}'
        files = sorted(f'{d}/{f}' for f in os.listdir(d) if f.endswith('_ml_predictions.csv'))
        parts = Parallel(n_jobs=N_JOBS)(delayed(read_cut)(f) for f in files)
        df = pd.concat([p for p in parts if p is not None], ignore_index=True)
        print(f'Sector {sector}: {len(df)} events from {len(files)} cuts in {(time.time() - start) / 60:.1f} min',
              flush=True)
        if SAVE_SLIM:
            df.to_csv(f'{OUT_DIR}/S{sector}_ml_slim.csv.gz', index=False, compression='gzip')
        sig5.append(df[df.lc_sig_max >= 5])
        texts.append(summary(df, sector))
        print(texts[-1], flush=True)
        samples.append(pick(df, sector, rng))

    pd.concat(sig5).to_csv(f'{OUT_DIR}/{name}_ml_sig5.csv.gz', index=False, compression='gzip')
    with open(f'{OUT_DIR}/{name}_ml_summary.txt', 'w') as f:
        f.write('\n\n'.join(texts))
    sample = pd.concat(samples).sample(frac=1, random_state=SEED)   # shuffled: strata and sectors mixed
    sample[KEY_COLS + ['frame_bin']].to_csv(f'{SAMPLE_DIR}/{name}_sort_sample.csv', index=False)
    sample.to_csv(f'{SAMPLE_DIR}/{name}_sort_sample_info.csv', index=False)
    print(f'\n{len(sample)} events to sort -> {SAMPLE_DIR}/{name}_sort_sample.csv')
    print(f'Copy down: {OUT_DIR}/{name}_ml_sig5.csv.gz, {OUT_DIR}/{name}_ml_summary.txt, '
          f'{SAMPLE_DIR}/{name}_sort_sample_info.csv')


def submit():
    """Write a batch script that runs this file with the same Python, and sbatch it."""
    log_dir = f'{OUT_DIR}/slurm_logs'
    os.makedirs(log_dir, exist_ok=True)
    script = f'{OUT_DIR}/sector_sample.sh'
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n'
                '#SBATCH --job-name=ml_sector_sample\n'
                f'#SBATCH --output={log_dir}/%j_out.txt\n'
                f'#SBATCH --error={log_dir}/%j_err.txt\n'
                '#SBATCH --ntasks=1\n'
                f'#SBATCH --cpus-per-task={N_JOBS}\n'
                f'#SBATCH --mem-per-cpu={MEM_PER_CPU_GB}G\n'
                f'#SBATCH --time={TIME}\n'
                f'#SBATCH --account={ACCOUNT}\n\n'
                'export PYTHONUNBUFFERED=1\n'
                f'{sys.executable} {os.path.abspath(__file__)}\n')
    result = subprocess.run(['sbatch', script], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'sbatch failed: {result.stderr.strip()}')
    print(result.stdout.strip())
    print(f'Log: {log_dir}/<jobid>_out.txt')


if __name__ == '__main__':
    if SLURM and 'SLURM_JOB_ID' not in os.environ:
        submit()
    else:
        main()
