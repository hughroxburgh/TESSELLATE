"""
How a stage-1 model classes the events the pipeline itself tagged (Asteroid / CosmicRay / Junk), runs locally.

With PIPELINE_WEIGHT = 0 the model never trained on these tags, so this is an independent check of each other:
where they agree, both are probably right; where they disagree, the events are worth a look (a sample of each
disagreement is written to OUT_DIR for plotting and review).

Edit the CONFIG block, then:  python ml_check_pipeline_tags.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # use this checkout's tessellate
from tessellate.ml_classifier import KEY_COLS, EventClassifier

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
MODEL = f'{HERE}/ml_data/event_classifier_stage1_pw0.joblib'
FEATURES = [f'{HERE}/ml_data/S27-55_training_features.csv.gz',
            f'{HERE}/ml_data/S27-55_training_features_pool.csv.gz']
TAGS = ['Asteroid', 'CosmicRay', 'Junk']
FRAME_BINS = [1]
SIG_BINS = [0, 5, 10, 20, np.inf]       # lc_sig_max bins for the breakdown
N_EXAMPLES = 50                         # per (tag, prediction) disagreement, saved for review
SEED = 1
OUT_DIR = f'{HERE}/ml_eval_pipeline_tags'
# ----------------


def main():
    cols = None
    parts = []
    for f in FEATURES:
        df = pd.read_csv(f, low_memory=False)
        parts.append(df[df.frame_bin.isin(FRAME_BINS) & df.classification.isin(TAGS)])
    tagged = pd.concat(parts, ignore_index=True).drop_duplicates(KEY_COLS).reset_index(drop=True)
    print(f'{len(tagged)} tagged frame-bin-{FRAME_BINS} events')

    clf = EventClassifier.load(MODEL)
    pred = clf.predict(tagged)
    tagged['pred'] = pred['pred_class'].to_numpy()
    for c in clf.classes_:
        tagged[f'p_{c}'] = pred[f'p_{c}'].to_numpy()
    tagged['sig_bin'] = pd.cut(tagged['tab_lc_sig_max'], SIG_BINS, right=False)

    out = [f'Pipeline tags vs {os.path.basename(MODEL)} (frame bins {FRAME_BINS})', '',
           'Rows = pipeline tag, columns = model prediction (fraction of the row; n = events):']
    table = pd.crosstab(tagged['classification'], tagged['pred'], normalize='index').round(3)
    table.insert(0, 'n', tagged['classification'].value_counts())
    out.append(table.to_string())
    for tag in TAGS:
        d = tagged[tagged.classification == tag]
        if not len(d):
            continue
        t = pd.crosstab(d['sig_bin'], d['pred'], normalize='index').round(3)
        t.insert(0, 'n', d['sig_bin'].value_counts().sort_index())
        out += ['', f'{tag} tags by lc_sig_max:', t.to_string()]
    text = '\n'.join(out) + '\n'

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f'{OUT_DIR}/report.txt', 'w') as f:
        f.write(text)
    rng = np.random.default_rng(SEED)
    keep = []
    for (tag, p), g in tagged[tagged.classification != tagged.pred].groupby(['classification', 'pred']):
        keep.append(g.iloc[rng.permutation(len(g))[:N_EXAMPLES]])
    if keep:
        ex = pd.concat(keep)
        ex[KEY_COLS + ['frame_bin', 'classification', 'pred', 'tab_lc_sig_max'] +
           [f'p_{c}' for c in clf.classes_]].to_csv(f'{OUT_DIR}/disagreements.csv', index=False)
    print(text)


if __name__ == '__main__':
    main()
