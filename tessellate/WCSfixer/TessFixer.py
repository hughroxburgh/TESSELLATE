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


class TessFixer():

    def __init__(self,sector,data_path='/fred/oz335/TESSdata'):

        self.sector = sector
        self.data_path = data_path

        self.path = f'{self.data_path}/Sector{self.sector}'

    def get_reference(self,cube_path,n_cuts=8):

        if not os.path.exists(cube_path):
            e = 'No cube file exists!'
            raise NameError(e)

        # -- Open data, and find notably bad frames -- #
        hdu = fits.open(cube_path)
        cube_data = hdu[1].data  # memmap, no RAM used yet

        HARD_BITMASK = 7407
        quality_mask = (hdu[2].data['DQUALITY'] & HARD_BITMASK) == 0
        bad = np.where(~quality_mask)[0]

        # bad = np.where(hdu[2].data['DQUALITY'] > 0)[0]
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

        done = False
        percentile = 5
        while not done:
            # -- Find frame where median is always in the lowest 5% -- #
            thresholds = np.percentile(medians, percentile, axis=0)  
            below = medians <= thresholds  
            below[bad] = False
            valid_mask = below.all(axis=1)  # shape: (3000,)
            valid_frames = np.where(valid_mask)[0]
            if len(valid_frames) > 0:
                candidate_medians = medians[valid_frames]          # (N, 64)
                score = candidate_medians.sum(axis=1)  # or .mean(axis=1)
                best_idx = valid_frames[np.argmin(score)]

                good_frames = np.where(quality_mask)[0]
                best_idx_good = int(np.searchsorted(good_frames, best_idx))

                print(f"Best frame index (all frames): {best_idx}")
                print(f"Best frame index (good frames only): {best_idx_good}")
                return best_idx, best_idx_good
            
            else:
                e = f"No frame has median in lowest {percentile}% across entire CCD."
                percentile += 1
        

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
            shutil.move(image_path, out)
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

                ref_idx_full, ref_idx_good = self.get_reference(f'{ccd_folder}/sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits')

                done = False
                while not done:
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
                        done=True
                    else:
                        ref_idx_full += 1
                        ref_idx_good += 1
                    
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



        
            