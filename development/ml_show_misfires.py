"""
Collect the images of the events a training run got wrong, to look through.

Reads a run's oof_predictions.csv (every sorted event, predicted by a model that
never saw its cut) and copies the PNG of each misclassified event from the sort
folders into RUN_DIR/misfires/<your label>_as_<prediction>/. File names start
with the model's probability for its prediction, so they sort by how sure it
was. Images are copied, never moved. Also writes misfires.csv.

Edit the CONFIG block, then:  python ml_show_misfires.py
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
RUN_DIR = f'{HERE}/ml_eval'                            # an OUT_DIR of ml_train_eval.py
SORT_DIRS = [f'{HERE}/S55/sort_found_flares', f'{HERE}/S55/sort_non_flares']
ARTEFACTS = ['Junk', 'CosmicRay', 'Systematic', 'Blend']
# ----------------


def main():
    oof = pd.read_csv(f'{RUN_DIR}/oof_predictions.csv', low_memory=False)
    oof = oof[oof.label_source == 'manual'].copy()
    classes = [c[2:] for c in oof if c.startswith('p_') and c != 'p_astrophysical']
    P = oof[[f'p_{c}' for c in classes]].to_numpy()
    oof['pred'] = np.array(classes)[P.argmax(axis=1)]
    oof['p_pred'] = P.max(axis=1)
    oof['p_real'] = oof[[f'p_{c}' for c in classes if c not in ARTEFACTS]].sum(axis=1)
    wrong = oof[oof.pred != oof.label].copy()

    images = {os.path.basename(p): p for d in SORT_DIRS for p in glob.glob(f'{d}/*/*.png')}
    out_dir = f'{RUN_DIR}/misfires'
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)          # only ever holds copies made by this script
    wrong['image'] = [f'S{r.sector}C{r.camera}C{r.ccd}C{r.cut}O{r.objid}E{r.eventid}.png' for r in wrong.itertuples()]
    for r in wrong.itertuples():
        src = images.get(r.image)
        if src is None:
            continue
        folder = f'{out_dir}/{r.label}_as_{r.pred}'
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(src, f'{folder}/{r.p_pred:.2f}_{r.image}')

    cols = ['label', 'pred', 'p_pred', 'p_real', 'image', 'sector', 'camera', 'ccd', 'cut', 'objid', 'eventid']
    wrong[cols].sort_values(['label', 'pred', 'p_pred'], ascending=[True, True, False]).to_csv(
        f'{RUN_DIR}/misfires.csv', index=False)
    print(f'{len(wrong)} of {len(oof)} sorted events misclassified -> {out_dir}')
    print(wrong.groupby(['label', 'pred']).size().rename('n').to_string())
    real_wrong = wrong[wrong.label.isin(ARTEFACTS) != wrong.pred.isin(ARTEFACTS)]
    print(f'\n{len(real_wrong)} of them cross the real/artefact line (the mistakes that matter most for filtering)')


if __name__ == '__main__':
    main()
