from tessellate import Navigator
from tqdm import tqdm
import pandas as pd
import os

project_path = '/fred/oz335/hroxburg/dev/final_localisation'
os.makedirs(f'{project_path}/images',exist_ok=True)

for sector in range(55,56):
    print('\n')
    events = pd.read_csv(f'{project_path}/non_flares.csv')
    for cam in range(1,2):
        for ccd in range(1,5):
            nav = Navigator(sector,cam,ccd)
            for cut in tqdm(range(1,65),desc=f'Sector {sector}, Camera {cam}, CCD {ccd}',position=0, leave=True,dynamic_ncols=False,ascii=True):
                cut_events = events[(events.camera==cam)&(events.ccd==ccd)&(events.cut==cut)]

                if len(cut_events) > 0:                    
                    for i,event in cut_events.iterrows():
                        if not os.path.exists(f'{project_path}/images/S{event.sector}C{event.camera}C{event.ccd}C{event.cut}O{event.objid}E{event.eventid}.png'):
                            
                            # nav.gather_results(cut=cut,sources=False,objects=False)
                            # nav.gather_data(cut=cut,verbose=False)
                            
                            nav.plot_lc(event,external_phot=True,save_combined_path=f'{project_path}/images',verbose=False)
