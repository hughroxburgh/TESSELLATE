from tessellate import Navigator
import pandas as pd
from tqdm import tqdm
import os
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import numpy as np

path = '/fred/oz335/hroxburg/dev/final_localisation'
os.makedirs(path,exist_ok=True)

for sector in range(55,56):
    print('\n')
    sector_df = pd.DataFrame()
    # os.makedirs(f'{path}/Sector{sector}',exist_ok=True)

    for cam in range(1,5):
        for ccd in range(1,5):
            nav = Navigator(sector,cam,ccd)
            for cut in tqdm(range(1,65),desc=f'Sector {sector}, Camera {cam}, CCD {ccd}',position=0, leave=True,dynamic_ncols=False,ascii=True):
                try:
                    nav.gather_results(cut=cut,sources=False)
                    evs = nav.filter_events(cut=cut,starkiller=False,asteroidkiller=True,frame_bin=1,max_frame_duration=40,
                                        lc_sig_max=10,centroid_err=0.3,flux_sign=1,psf_like=0.75,cosmicraykiller=True,
                                        boundary_buffer=20,lc_flat='hard',exclude_bad_frames=True)#,min_frame_duration=3)
                    
                except:
                    print(f'Cam {cam} CCC {ccd} Cut {cut} does not exist!')
                    continue
                

                # -- Remove Asteroids -- #
                mask = (
                    (evs['gaussian_score'] + evs['com_motion']*1.4 >= 1.3)
                )

                evs = evs[~mask]


                # # -- Check for deeper stars -- #
                # if len(evs) > 0:

                #     ping = False
                #     matched_idx = []
                #     gaia_local = pd.read_csv(f'/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/local_gaia_cat.csv')
                
                #     for i,row in evs.iterrows():
                #         coord = SkyCoord(ra=row.ra*u.degree,dec=row.dec*u.degree)
                #         max_rad = max(row.ra_err * 3600 * 3,row.dec_err * 3600 * 3)

                #         v = Vizier(columns=["RA_ICRS",'DE_ICRS','Gmag','Source'])
                #         results = v.query_region(coord, radius=max_rad*u.arcsec, catalog="I/355/gaiadr3",cache=False)
                #         if len(results) > 0:
                #             results = results[0].to_pandas()
                #             inside = results[(abs(results.RA_ICRS-row.ra)<3*row.ra_err)&(abs(results.DE_ICRS-row.dec)<3*row.dec_err)]
                #             if len(inside) > 0:

                #                 source = inside.sort_values('Gmag').iloc[0]
                #                 star = {'Source':source.Source,'mag': source.Gmag-0.5,'ra':source.RA_ICRS,'dec':source.DE_ICRS}
                                
                #                 if not np.isin(star['Source'],gaia_local.Source):
                #                     gaia_local.loc[len(gaia_local)] = star

                #                 nav.events.loc[(nav.events['objid'] == row.objid) & (nav.events['eventid'] == row.eventid),
                #                                 'gaia_id'] = star['Source']

                #                 nav.objects.loc[nav.objects['objid']==row.objid,'gaia_id'] = star['Source']
                            
                #                 matched_idx.append(i)
                #                 ping = True
                #                 print('Fixed!')
                
                #     if ping:
                #         gaia_local.to_csv(f'/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/local_gaia_cat.csv',index=False)
                #         nav.events.to_csv(f'/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/detected_events.csv',index=False)
                #         nav.objects.to_csv(f'/fred/oz335/TESSdata/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of64/detected_objects.csv',index=False)
                
                #     evs = evs.drop(index=matched_idx)
                sector_df = pd.concat([sector_df,evs])
           
    sector_df.to_csv(f'{path}/found_flares.csv',index=False)

    

            # d = Detector(sector=sector, cam=cam, ccd=ccd)

            # d.collate_filtered_events(save_path=f'/fred/oz335/projects/highlat_transients/sig10maxevents5/Sector{sector}',
            #                     lower=2,upper=40,min_events=1,max_events=5,psf_like=0.85,lc_sig_max=10,flux_sign=1,
            #                     asteroidkiller=True,galactic_latitude=15.,boundarykiller=True, centroid_err=0.1) #,density_score=5)

# for cam in range(1,5):
#     for ccd in range(1,5):
#         print('\n')
#         print(f'Sector {sector}, Camera {cam}, CCD {ccd}')
#         d = Detector(sector=sector, cam=cam, ccd=ccd,n=4,data_path='/fred/oz335/TESSdata')

#         d.plot_filtered_events(save_path=f'/fred/oz335/projects/highlat_transients/events_sig10maxevents1/Complete/Sector{sector}/Manual/Interesting',tess_grid=3)        
