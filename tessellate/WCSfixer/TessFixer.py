import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
from time import time 
from tessellate import DataProcessor
from glob import glob
import shutil
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from .wcs_fix import Calculate_WCS,Update_SIP_Header,Final_Shift_Update
from .fit_stars import Fit_CCD


HARD_BITMASK = 7407     # lightkurve's 'hard' quality bitmask, as TESSreduce applies to the cuts
NOT_REFERENCE_BITS = 16384  # BadCalibrationExclude: kept by the hard bitmask (so still counted in GoodIdx) but never a
                            # reference -- such FFIs can be blank or lack a celestial WCS (S42 Cam4 Ccd1/2, 2026-10-22)


def rank_reference_frames(medians, quality_mask, start_percentile=5, not_reference=None):
    """
    Reference frame candidates, best first, as (full index, index among quality-good frames).

    medians: (frames, blocks) median of each block of the CCD in each frame; quality_mask: True = frame passes the
    hard bitmask; not_reference: optional True = never a candidate (frames flagged NOT_REFERENCE_BITS). Blank frames
    (a block median <= 0 or not finite: no data, e.g. at orbit edges, where they can carry only flag 16384, which the
    hard bitmask keeps) are never candidates either -- a median of 0 is the lowest everywhere.
    A candidate has every block's median within the lowest p% of the usable frames; p starts at start_percentile and
    rises until frames qualify. Candidates at one p are ordered by the sum of their block medians; further p values
    add the frames that newly qualify, so a caller can keep asking.
    The good-frame index is the frame's position among quality-good frames -- what TESSreduce's ref_ind indexes
    (lightkurve drops flagged frames), and only defined for a quality-good frame, which every candidate is.
    """
    medians = np.asarray(medians, float)
    quality_mask = np.asarray(quality_mask, bool)
    blank = ~np.isfinite(medians).all(axis=1) | (medians <= 0).any(axis=1)
    usable = quality_mask & ~blank
    if not_reference is not None:
        usable &= ~np.asarray(not_reference, bool)
    if not usable.any():
        raise ValueError('No usable reference frame: every frame is flagged or blank')
    good_frames = np.flatnonzero(quality_mask)
    score = np.where(usable, np.nansum(medians, axis=1), np.inf)
    seen = set()
    for percentile in range(start_percentile, 101):
        thresholds = np.percentile(medians[usable], percentile, axis=0)
        valid = np.flatnonzero(usable & (medians <= thresholds).all(axis=1))
        for idx in valid[np.argsort(score[valid], kind='stable')]:
            if idx not in seen:
                seen.add(idx)
                yield int(idx), int(np.searchsorted(good_frames, idx))


class TessFixer():

    def __init__(self,sector,data_path='/fred/oz335/TESSdata'):

        self.sector = sector
        self.data_path = data_path

        self.path = f'{self.data_path}/Sector{self.sector}'

    def get_reference(self,cube_path,n_cuts=8):
        """(full index, index among quality-good frames) of the best reference frame (see reference_candidates)."""
        return next(self.reference_candidates(cube_path,n_cuts))

    def reference_candidates(self,cube_path,n_cuts=8):
        """
        Reference frame candidates for a CCD, best first, as (full index, index among quality-good frames).
        Medians of the n_cuts x n_cuts blocks of every frame are ranked by rank_reference_frames.
        cube_path may instead be the frame-stats file (.npz) written by DataProcessor.make_direct_cuts, which
        holds those medians already -- no cube needed.
        """

        if not os.path.exists(cube_path):
            e = 'No cube file exists!'
            raise NameError(e)

        if cube_path.endswith('.npz'):
            stats = np.load(cube_path)
            if int(stats['blocks']) != n_cuts:
                raise ValueError(f'{cube_path} has {int(stats["blocks"])}x{int(stats["blocks"])} block medians, '
                                 f'not {n_cuts}x{n_cuts}')
            dquality = np.asarray(stats['dquality'], int)
            quality_mask = (dquality & HARD_BITMASK) == 0
            yield from rank_reference_frames(stats['medians'], quality_mask,
                                             not_reference=(dquality & NOT_REFERENCE_BITS) != 0)
            return

        # -- Open data, and find notably bad frames -- #
        hdu = fits.open(cube_path)
        cube_data = hdu[1].data  # memmap, no RAM used yet

        dquality = np.asarray(hdu[2].data['DQUALITY'], int)
        quality_mask = (dquality & HARD_BITMASK) == 0

        print('loading data',end='\r')
        ts = time()
        flux = np.array(cube_data[:,:,:,0])
        flux = np.transpose(flux, (2, 0, 1))
        print(f'loading data -- done! ({time()-ts:.0f}s)')

        # -- Calculate rolling median of cuts across the CCD -- #
        print(f'calculating {n_cuts**2} medians',end='\r')
        size = int(2048/n_cuts)
        medians = []
        for i in range(n_cuts):
            for j in range(n_cuts):
                cut = flux[:,size*i:size*(i+1),44+size*j:44+size*(j+1)]
                median = np.nanmedian(cut,axis=(1,2))
                medians.append(median)
        print(f'calculating {n_cuts**2} medians -- done! ({time()-ts:.0f}s)')

        medians = np.array(medians).T
        del flux

        for best_idx, best_idx_good in rank_reference_frames(medians, quality_mask,
                                                             not_reference=(dquality & NOT_REFERENCE_BITS) != 0):
            print(f"Best frame index (all frames): {best_idx}")
            print(f"Best frame index (good frames only): {best_idx_good}")
            yield best_idx, best_idx_good

    def load_gaia(self,cam,ccd,image_path,gaia_path='/fred/oz335/GAIAdata/full_gaia_cat.csv'):

        ts = time()
        print('Getting gaia cat',end='\r')
        ccd_gaia_path = f'{self.path}/Cam{cam}/Ccd{ccd}/ccd_gaia_cat.csv'
        if os.path.exists(ccd_gaia_path):
            gaia = pd.read_csv(ccd_gaia_path)
            
        else:

            import duckdb

            with fits.open(image_path) as f:
                wcs = WCS(f[1].header)

            # gaia = pd.read_csv(gaia_path)
            centre_ra, centre_dec = wcs.all_pix2world(1068, 1024, 0)
            
            radius_arcsec = 32000  # max radius = sqrt(1024**2 * 2)px * 21"

            gaia = duckdb.sql(f"""
                SELECT * FROM (
                    SELECT *,
                        2 * degrees(asin(sqrt(
                            pow(sin(radians(("dec" - {centre_dec}) / 2)), 2) +
                            cos(radians("dec")) * cos(radians({centre_dec})) *
                            pow(sin(radians((ra - {centre_ra}) / 2)), 2)
                        ))) * 3600 AS dist_arcsec
                    FROM read_csv('{gaia_path}', ignore_errors=true)
                )
                WHERE dist_arcsec < {radius_arcsec}
            """).df()
            
            gaia['x'],gaia['y'] =  wcs.all_world2pix(gaia.ra, gaia.dec, 0)
            gaia = gaia[(gaia.x>=40)&(gaia.x<=2096)&(gaia.y>=-4)&(gaia.y<=2052)]
            gaia = gaia.drop(columns=['dist_arcsec','x','y'])
            gaia.rename(columns={'source_id':'Source'},inplace=True)
            gaia.to_csv(ccd_gaia_path,index=False)

        print(f'Getting gaia cat -- done! ({time()-ts:.0f}s)')

        return gaia[(gaia.RPmag<16)&(gaia.RPmag >9)]

        
    def fix_wcs(self,ccd_folder,image_path,gaia_cat,cam,ccd,num,ref=False,order=6):

        # -- Make save folder and move image -- #
        save_folder = f'{ccd_folder}/wcs/ref' if ref else f'{ccd_folder}/wcs/im{num}' 
        out = f'{save_folder}/corrected.fits'
        if not os.path.exists(save_folder): 
            os.mkdir(save_folder) 
            shutil.copy(image_path, out)   # copy, not move: image_files stays complete, so frame indices hold
            shutil.copy(out, out.replace('corrected.fits', 'old.fits'))
            if ref:
                with open(f'{save_folder}/reference.txt', 'w') as file: 
                    file.write(f'FullIdx: {num[0]}, GoodIdx: {num[1]}')
                               
        # -- Fit stars -- #
        if not os.path.exists(f'{save_folder}/ccd_sourcefits.csv'):
            stars = Fit_CCD(image_path=out,sector=self.sector,cam=cam,ccd=ccd,gaia_cat=gaia_cat,n=4,quality_thresh=0.75)
            stars.to_csv(f'{save_folder}/ccd_sourcefits.csv',index=False)
            
        # -- Calculate and update WCS -- #
        Calculate_WCS(save_folder,order)
        Update_SIP_Header(save_folder,order)
        Final_Shift_Update(save_folder)


    def run(self,cam,ccd,all=False):

        ccd_folder = f'{self.path}/Cam{cam}/Ccd{ccd}'

        if not os.path.exists(f'{ccd_folder}/wcs'):
            os.mkdir(f'{ccd_folder}/wcs')

        # -- Load gaia catalogue for this CCD -- #

        data_processor = DataProcessor(sector=self.sector,data_path=self.data_path)
        if not all:
            if not os.path.exists(f"{ccd_folder}/wcs/ref/corrected.fits"):

                # -- Best candidate whose FFI has a celestial WCS. The next candidate is the next-best quiet,
                # -- quality-good frame (never just the next frame in time, which may be flagged or blank), and
                # -- both indices come from the candidate itself, so GoodIdx can't drift.
                # -- frame stats from direct cutting if there are any, else the cube -- #
                cube_path = f'{ccd_folder}/sector{self.sector}_cam{cam}_ccd{ccd}_frame_stats.npz'
                if not os.path.exists(cube_path):
                    cube_path = f'{ccd_folder}/sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
                for ref_idx_full, ref_idx_good in self.reference_candidates(cube_path):
                    if not os.path.exists(f'{ccd_folder}/image_files/'):
                        data_processor.download(cam=cam,ccd=ccd,single=ref_idx_full,number=None)
                        image_path = glob(f'{ccd_folder}/image_files/*.fits')[0]
                    elif len(glob(f'{ccd_folder}/image_files/*.fits')) < 100:
                        os.system(f'rm -r {ccd_folder}/image_files')
                        data_processor.download(cam=cam,ccd=ccd,single=ref_idx_full,number=None)
                        image_path = glob(f'{ccd_folder}/image_files/*.fits')[0]
                    else:
                        image_path = sorted(glob(f'{ccd_folder}/image_files/*.fits'))[ref_idx_full]

                    with fits.open(image_path) as f:
                        wcs = WCS(f[1].header)
                    if wcs.is_celestial:
                        break
                    print(f'Frame {ref_idx_full} has no celestial WCS, trying the next candidate')
                else:
                    raise RuntimeError(f'No reference candidate with a celestial WCS for Cam {cam} Ccd {ccd}')

            else:
                image_path = f'{ccd_folder}/wcs/ref/corrected.fits'
                with open(f'{ccd_folder}/wcs/ref/reference.txt', 'r') as file:
                    line = file.readline()
                    parts = line.split(',')
                    ref_idx_full = int(parts[0].split(':')[1].strip())
                    ref_idx_good = int(parts[1].split(':')[1].strip())

            gaia_cat = self.load_gaia(cam,ccd,image_path)
            self.fix_wcs(ccd_folder,image_path,gaia_cat,cam,ccd,num=[ref_idx_full, ref_idx_good],ref=True)
            
        else:
            data_processor.download(cam=cam,ccd=ccd)
            images = sorted(glob(f'{ccd_folder}/image_files'))
            gaia_cat = self.load_gaia(images[100])
            for i,image_path in enumerate(images):
                self.fix_wcs(ccd_folder,image_path,gaia_cat,cam,ccd,num=i)



        
            