"""
Collect every manual sort into one label table for the ML classifier, and print counts.

A sort folder holds one sub-folder per group (Flare, Variable, Asteroid, Junk, ...), each with
the sorted PNGs (S{s}C{cam}C{ccd}C{cut}O{objid}E{eventid}.png) and/or an events.csv of their rows.
An event counts as sorted into a group if its PNG is there OR its row is in that events.csv.
Events sorted into more than one group are dropped (listed in the conflicts csv).

Output columns:
  sector, camera, ccd, cut, objid, eventid   the event key
  label        the fine class (the ml_classifier CLASSES names: Junk, CosmicRay, Flare, ...)
  stage1       Artefact / Asteroid / Flare / Variable (Interesting -> Flare; see STAGE1)

Folders in EXCLUDE_GROUPS (currently Interesting) are left out entirely.
  source       which SORTS entry it came from (its selection defines the label's domain)
  group        the original folder name
  + frame_bin, xcentroid, ycentroid, mjd_max when the events.csv has them (for re-matching
    after a detection re-run)

Uses pandas only (no tessellate import), so it runs with any installed branch.
"""
import glob
import os
import re

import pandas as pd

# ---- CONFIG ----
# SOURCE name -> list of sort folders (glob patterns allowed). Each sort folder contains the group folders.
SORTS = {
    # sig10 off-star sorts: filter_events(starkiller='hard', lc_sig_max=10, centroid_err=0.1 (old 95% radius),
    # psf_like=0.75, frame_bin=1, max_frame_duration=40, flux_sign=1, |b| > 15, asteroid/CR killers), old run
    'highlat_sig10': ['/fred/oz335/projects/highlat_transients/sig10/Sector*'],
    # S55 sorts (found_flares = Gaia-matched list, non_flares = the rest); edit to the cluster paths
    # 'S55_found_flares': ['/path/to/sort_found_flares'],
    # 'S55_non_flares': ['/path/to/sort_non_flares'],
}

# Folder name -> fine class. Folders not listed here are reported and left out.
GROUP_TO_LABEL = {
    'Flare': 'Flare', 'Flares': 'Flare',
    'Variable': 'Variable', 'Variables': 'Variable',
    'Asteroid': 'Asteroid', 'Asteroids': 'Asteroid',
    'CosmicRay': 'CosmicRay', 'Cosmic Ray': 'CosmicRay', 'Cosmic_Ray': 'CosmicRay',
    'Junk': 'Junk', 'Systematic': 'Systematic', 'Blend': 'Blend',
    'Interesting': 'Interesting', 'Other': 'Interesting',
}

# Folders left out on purpose (not reported as unknown). Interesting: not pure enough yet (Hugh, 2026-09-28);
# remove it from this list to bring those events back (as stage1 Flare, see STAGE1).
EXCLUDE_GROUPS = ['Interesting', 'Other']

# Fine class -> stage-1 class. Interesting = flare-like events that may be extragalactic transients: Flare-like
# in stage 1; the fine label keeps them apart as stage-2 candidates.
STAGE1 = {
    'Junk': 'Artefact', 'CosmicRay': 'Artefact', 'Systematic': 'Artefact', 'Blend': 'Artefact',
    'Asteroid': 'Asteroid', 'Flare': 'Flare', 'Variable': 'Variable', 'Interesting': 'Flare',
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_labels')
OUT_NAME = 'manual_labels.csv'
# ---- END CONFIG ----

KEY_COLS = ['sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
EXTRA_COLS = ['frame_bin', 'xcentroid', 'ycentroid', 'mjd_max']
IMAGE_NAME = re.compile(r'S(\d+)C(\d+)C(\d+)C(\d+)O(\d+)E(\d+)')


def read_sort_folder(sort_dir, source):
    """Rows (key, group, source) from one sort folder, plus the extra columns where available."""
    rows, info, skipped = [], [], set()
    for group in sorted(os.listdir(sort_dir)):
        gdir = os.path.join(sort_dir, group)
        if not os.path.isdir(gdir) or group in EXCLUDE_GROUPS:
            continue
        if group not in GROUP_TO_LABEL:
            skipped.add(group)
            continue
        for name in os.listdir(gdir):
            m = IMAGE_NAME.match(name)
            if m and name.lower().endswith('.png'):
                rows.append(dict(zip(KEY_COLS, map(int, m.groups())), group=group))
        csv = os.path.join(gdir, 'events.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            if set(KEY_COLS) <= set(df):
                df = df[KEY_COLS + [c for c in EXTRA_COLS if c in df]].copy()
                for c in KEY_COLS:
                    df[c] = df[c].astype(int)
                rows.extend(dict(r, group=group) for r in df[KEY_COLS].to_dict('records'))
                info.append(df)
            else:
                print(f'  {csv}: missing key columns {sorted(set(KEY_COLS) - set(df))}; using PNG names only')
    labels = pd.DataFrame(rows, columns=KEY_COLS + ['group']).drop_duplicates()
    labels['source'] = source
    info = pd.concat(info, ignore_index=True).drop_duplicates(KEY_COLS) if info else None
    return labels, info, skipped


def main():
    all_labels, all_info, skipped = [], [], set()
    for source, patterns in SORTS.items():
        dirs = sorted(d for p in patterns for d in glob.glob(p) if os.path.isdir(d))
        if not dirs:
            print(f'{source}: no folders match {patterns}')
            continue
        for d in dirs:
            labels, info, sk = read_sort_folder(d, source)
            print(f'{source}: {d}: {len(labels)} sorted events')
            all_labels.append(labels)
            skipped |= {f'{source}/{g}' for g in sk}
            if info is not None:
                all_info.append(info)
    if not all_labels:
        raise SystemExit('No sort folders found; check SORTS.')
    if skipped:
        print(f'\nFolders not in GROUP_TO_LABEL (left out): {sorted(skipped)}')

    labels = pd.concat(all_labels, ignore_index=True)
    labels['label'] = labels['group'].map(GROUP_TO_LABEL)
    labels = labels.drop_duplicates(KEY_COLS + ['label'])

    # the same event in two different classes (within or across sources): drop, and save the list
    conflict = labels.duplicated(KEY_COLS, keep=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    if conflict.any():
        conflicts = labels[conflict].sort_values(KEY_COLS)
        conflicts.to_csv(os.path.join(OUT_DIR, OUT_NAME.replace('.csv', '_conflicts.csv')), index=False)
        print(f'\n{conflicts[KEY_COLS].drop_duplicates().shape[0]} events sorted into more than one class: dropped '
              f'(see {OUT_NAME.replace(".csv", "_conflicts.csv")})')
        labels = labels[~conflict]

    labels['stage1'] = labels['label'].map(STAGE1)
    if all_info:
        info = pd.concat(all_info, ignore_index=True).drop_duplicates(KEY_COLS)
        labels = labels.merge(info, on=KEY_COLS, how='left')
    labels = labels.sort_values(KEY_COLS).reset_index(drop=True)

    out = os.path.join(OUT_DIR, OUT_NAME)
    labels.to_csv(out, index=False)
    print(f'\nWrote {len(labels)} labelled events to {out}')

    print('\nStage-1 class by source:')
    print(pd.crosstab(labels['source'], labels['stage1'], margins=True).to_string())
    print('\nFine class by sector:')
    print(pd.crosstab(labels['sector'], labels['label'], margins=True).to_string())
    missing = labels['xcentroid'].isna().sum() if 'xcentroid' in labels else len(labels)
    if missing:
        print(f'\n{missing} events have no position/time from an events.csv (PNG only): '
              f'they can only be matched by objid/eventid, so need the same detection run.')


if __name__ == '__main__':
    main()
