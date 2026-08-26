import os
import numpy as np
from time import time as t

from .tools import _Save_space, _Remove_emptys, _Extract_fits, _Print_buff, save_compact_array, save_table, load_table, table_exists
    
# def _get_wcs(path,wcs_path_check,verbose=1):
#     """
#     Get WCS data from a file in the path
#     """

#     import shutil
#     from glob import glob
#     from astropy.wcs import WCS

#     if os.path.exists(wcs_path_check):
#         wcsFile = _Extract_fits(wcs_path_check)
#         wcsItem = WCS(wcsFile[1].header)
#     else:
#         if glob(f'{path}/*ffic.fits'):
#             done = False
#             i = 0
#             while not done:
#                 filepath = glob(f'{path}/*ffic.fits')[i]
#                 file = _Extract_fits(filepath)
#                 wcsItem = WCS(file[1].header)
#                 file.close()
#                 if wcsItem.get_axis_types()[0]['coordinate_type'] == 'celestial':
#                     done = True
#                     shutil.copy2(filepath,wcs_path_check)
#                 else:
#                     i += 1
#         else:
#             if verbose>0:
#                 print('No Data!')
#             return

#     return wcsItem

def _cut_properties(wcsItem,n,overlap): 

        intervals = 2048/n

        cutCornersX = [44 + i*intervals for i in range(n)]
        cutCornersY = [i*intervals for i in range(n)]
        cutCorners = np.meshgrid(cutCornersX,cutCornersY)
        cutCorners = np.floor(np.stack((cutCorners[0],cutCorners[1]),axis=2).reshape(n**2,2))

        idx = np.arange(n * n)
        cols = idx % n
        rows = idx // n
        cutCorners[cols == n - 1, 0] -= overlap
        cutCorners[(cols != 0) & (cols != n - 1), 0] -= overlap // 2
        cutCorners[rows == n - 1, 1] -= overlap
        cutCorners[(rows != 0) & (rows != n - 1), 1] -= overlap // 2

        intervals = np.ceil(intervals)
        rad = np.ceil(intervals / 2) + overlap//2

        cutCentrePx = cutCorners + rad
        cutCentreCoords = np.array(wcsItem.all_pix2world(cutCentrePx[:,0],cutCentrePx[:,1],0)).transpose()

        return cutCorners,cutCentrePx,cutCentreCoords,rad

def _parallel_cuts(sector,cam,ccd,cut,n,cube_path,file_path,coords,size,verbose):

    from astrocut import CutoutFactory

    name = f'sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
    if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
        print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
    else:
        if verbose > 0:
            print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
        
        my_cutter = CutoutFactory() # astrocut class
        _Save_space(f'{file_path}/Cut{cut}of{n**2}')

        # -- Cut -- #
        cut_file = my_cutter.cube_cut(cube_path, 
                                        f"{coords[0]} {coords[1]}", 
                                        (size,size), 
                                        output_path = f'{file_path}/Cut{cut}of{n**2}',
                                        target_pixel_file = name,
                                        verbose=(verbose>1)) 

        if verbose > 0:
            print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
            print('\n')

class DataProcessor():

    def __init__(self,sector,verbose=1,data_path=None) -> None:

        self.sector = sector
        self.verbose = verbose

        if data_path[-1] == '/':
            data_path = data_path[:-1]
        self.data_path = data_path

        self._make_path(False)

    def _make_path(self,delete):
        """
        Creates a folder for the path. 
        """

        if self.data_path is None:
            _Save_space('temporary',delete=delete)
            self.path = './temporary'
        else:
            _Save_space(f'{self.data_path}/Sector{self.sector}')
            self.path = f'{self.data_path}/Sector{self.sector}'

    def download(self,cam,ccd,number='all',time=None,single=None):
        """
        Function for downloading FFIs from MAST archive.

        ------
        Inputs
        ------
        cam : int
            specific camera, default None
        ccd : int
            desired ccd, default None

        -------
        Options:
        -------
        number : int
            if not None, downloads this many
        time : float (MJD)
            if not None, downloads only FFIs within a day of this time
        
        """

        from .downloader import Download_cam_ccd_FFIs
        
        if self.verbose > 0:
            print(_Print_buff(50,f'Downloading Sector {self.sector} Cam {cam} CCD {ccd}'))
        Download_cam_ccd_FFIs(self.path,self.sector,cam,ccd,time,None,None,number=number,single=single) 
    
    def find_cuts(self,cam,ccd,n,overlap=10,plot=True,proj=True,coord=None,verbose=1):
        """
        Function for finding cuts.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts will be made

        -------
        Options:
        -------
        plot : bool
            if True, plot cuts 
        proj : bool
            if True, plot cuts with WCS grid
        coord : (ra,dec)
            if not None, plots coord 
        
        """

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from astropy.wcs import WCS
        from astropy.io import fits
        from glob import glob

        newpath = f'{self.path}/Cam{cam}/Ccd{ccd}'

        # wcsItem = WCS(fits.open(f'{newpath}/wcs/ref/corrected.fits')[1].header)

        if not os.path.exists(f'{newpath}/wcs/ref/corrected.fits'):
            if len(glob(f'{newpath}/image_files/*ffic.fits')) == 0:
                data_processor = DataProcessor(sector=self.sector,data_path=self.data_path,verbose=0)
                data_processor.download(cam=cam,ccd=ccd,number=1)

            file = glob(f'{newpath}/image_files/*ffic.fits')[0]
            wcsItem = WCS(fits.open(file)[1].header)

        else:
            wcsItem = WCS(fits.open(f'{newpath}/wcs/ref/corrected.fits')[1].header)

        # wcsItem = _get_wcs(f'{newpath}/image_files',f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits') 
        # if not os.path.exists(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits'):
        #     wcs_save = wcsItem.to_fits()
        #     wcs_save.writeto(f'{newpath}/sector{self.sector}_cam{cam}_ccd{ccd}_wcs.fits')

        if wcsItem is None:
            if verbose > 0:
                print('WCS Extraction Failed')
            return

        cutCorners, cutCentrePx, cutCentreCoords, cutSize = _cut_properties(wcsItem,n,overlap)

        if plot:
            # -- Plots data -- #
            fig = plt.figure(constrained_layout=False, figsize=(6,6))
            
            if proj:
                ax = plt.subplot(projection=wcsItem)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
            else:
                ax = plt.subplot()

            if coord is not None:
                coordPx = wcsItem.all_world2pix(coord[0],coord[1],0)
                ax.scatter(coordPx[0],coordPx[1],s=10)
                ax.text(x=coordPx[0],y=coordPx[1]+10,s=f'{coordPx[0]:.1f},{coordPx[1]:.1f}')
            
            # -- Real rectangle edge -- #
            rectangleTotal = patches.Rectangle((44,0), 2048, 2048,edgecolor='black',facecolor='none',alpha=0.5)
            
            # -- Sets title -- #
            ax.set_title(f'Camera {cam} CCD {ccd}')
            ax.set_xlim(0,2136)
            ax.set_ylim(0,2078)
            ax.grid()

            ax.add_patch(rectangleTotal)
                
            # -- Adds cuts -- #
            colours = iter(plt.cm.rainbow(np.linspace(0, 1, n**2)))

            for corner in cutCorners:
                c = next(colours)
                rectangle = patches.Rectangle(corner,2*cutSize,2*cutSize,edgecolor=c,
                                              facecolor='none',alpha=1)
                ax.add_patch(rectangle)
                
        return cutCorners, cutCentrePx, cutCentreCoords, cutSize
    
    def _make_part_cube(self,cam,ccd,input_files):

        from astrocut import CubeFactory
        from astropy.time import Time
        import datetime

        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        
        dates = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            dates.append(imagetime)

        sortedDates = np.array(sorted(dates))
        differences = np.diff(sortedDates)
        idx = np.where(differences == np.nanmax(differences))[0][0]+1

        cube1_files = []
        cube2_files = []
        for f in input_files:
            year = f[56:60]
            daynum = f[60:63]
            hour = f[63:65]
            minute = f[65:67]
            sec = f[67:69]
            date = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j")
            month = date.month
            day = date.day
            imagetime = '{}-{}-{}T{}:{}:{}'.format(year,month,day,hour,minute,sec)
            imagetime = Time(imagetime, format='isot', scale='utc').mjd
            if imagetime in sortedDates[:idx]:
                cube1_files.append(f)
            elif imagetime in sortedDates[idx:]:
                cube2_files.append(f)  
        cubes = [cube1_files,cube2_files]

        for i in range(2):
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            if not os.path.exists(f'{broad_path}/Part{i+1}'):
                os.mkdir(f'{broad_path}/Part{i+1}')
            cube_path = f'{broad_path}/Part{i+1}/{cube_name}'

            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd} Part {i+1}'))

            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(cubes[i],cube_file=cube_path,verbose=self.verbose>1,max_memory=500)
            print('\n')


    def make_cube(self,cam,ccd,part=False):
        """
        Make cube for this cam,ccd.
        
        ------
        Inputs
        ------
        cam : int
            desired camera 
        ccd : int
            desired ccd

        -------
        Options
        -------
        delete_files : bool  
            deletes all FITS files once cube is made
        cubing_message : str
            custom printout message for self.verbose > 0

        -------
        Creates
        -------
        Data cube fits file in path.

        """

        from astrocut import CubeFactory
        from glob import glob


        # -- Generate Cube Path -- #
        broad_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        file_path = f'{broad_path}/image_files'

        # if os.path.exists(cube_path):
        #     print(f'Cam {cam} CCD {ccd} cube already exists!')
        #     return

        input_files = glob(f'{file_path}/*ffic.fits')  # list of fits files in path
        if len(input_files) < 1:
            print('No files to cube!')
            return  
        
        deleted = _Remove_emptys(input_files)  # remove empty downloaded fits files
        if self.verbose > 1:
            print(f'Deleted {deleted} corrupted file/s.')
                    
        input_files = glob(f'{file_path}/*ffic.fits')  # regather list of good fits files
        if len(input_files) < 1:
            print('No files to cube!')
            return    

        if self.verbose > 1:
            print(f'Number of files to cube = {len(input_files)}')
            size = len(input_files) * 0.0355
            print(f'Estimated cube size = {size:.2f} GB')

        if part:
            self._make_part_cube(cam,ccd,input_files)
        else:
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{broad_path}/{cube_name}'

            # -- Allows for a custom cubing message (kinda dumb) -- #
            if self.verbose > 0:
                print(_Print_buff(50,f'Cubing Sector {self.sector} Cam {cam} CCD {ccd}'))
            
            # -- Make Cube -- #
            cube_maker = CubeFactory()
            cube_file = cube_maker.make_cube(input_files,cube_file=cube_path,verbose=self.verbose>1,max_memory=500)

    def _make_part_cuts(self,cam,ccd,n,cut,file_path,cutCentreCoords, cutSize):

        from astrocut import CutoutFactory

        for i in range(2):

            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/Part{i+1}/{cube_name}'
        
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} Cut {cut} (of {n**2}) Part {i+1}')
        
            my_cutter = CutoutFactory() # astrocut class
            coords = cutCentreCoords[cut-1]

            _Save_space(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}')
                        
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                f"{coords[0]} {coords[1]}", 
                                                (cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} Cut {cut} Part {i+1} complete.')
                print('\n')

            with open(f'{file_path}/Part{i+1}/Cut{cut}of{n**2}/cut.txt', 'w') as file:
                file.write('Cut!')
    
    def make_cuts(self,cam,ccd,n,cut,part=False):
        """
        Make cut(s) for this CCD.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd

        ------
        Creates
        ------
        Save files for cut(s) in path.

        """

        from astrocut import CutoutFactory

        try:
            _, cutCentrePx, _, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False)
        except:
            print('Something wrong with finding cuts!')
            return
        
        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        if not os.path.exists(file_path):
            print('No data to cut!')
            return
        
        if part:
            self._make_part_cuts(cam,ccd,n,cut,file_path,cutCentrePx,cutSize)
        else:   
            # -- Generate Cube Path -- #
            cube_name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cube.fits'
            cube_path = f'{file_path}/{cube_name}'
            
            name = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            # if os.path.exists(f'{file_path}/Cut{cut}of{n**2}/{name}'):
            #     print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
            # else:
            if self.verbose > 0:
                print(f'Cutting Cam {cam} CCD {ccd} cut {cut} (of {n**2})')
            
            my_cutter = CutoutFactory() # astrocut class
            px = cutCentrePx[cut-1]

            _Save_space(f'{file_path}/Cut{cut}of{n**2}')
                            
            # -- Cut -- #
            self.cut_file = my_cutter.cube_cut(cube_path, 
                                                xy_pos=(px[0],px[1]), 
                                                cutout_size=(cutSize*2,cutSize*2), 
                                                output_path = f'{file_path}/Cut{cut}of{n**2}',
                                                target_pixel_file = name,
                                                verbose=(self.verbose>1)) 

            if self.verbose > 0:
                print(f'Cam {cam} CCD {ccd} cut {cut} complete.')
                print('\n')

    def predict_asteroids(self,cam,ccd,n,cut,part=False):
        """
        Predicts every catalogued minor planet (from MPCORB) crossing this
        cut's footprint at any point during its observing window, using
        ASSIST-perturbed orbit propagation (see asteroid_prediction.py).
        Runs ahead of reduce() -- only needs the cut's sky footprint (from
        find_cuts) and frame times, both already available once make_cuts()
        has produced the cut's TPF (Times.npy isn't written until reduce()).

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts have been made
        cut : int
            specific cut

        ------
        Creates
        ------
        Per-cut asteroid ephemeris CSV and trail plot in the cut's folder.

        """

        import lightkurve as lk
        from astropy.io import fits
        from astropy.wcs import WCS
        from .asteroid_prediction import predict_asteroids_for_footprint, plot_asteroid_trails

        try:
            cutCorners, _, cutCentreCoords, cutSize = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)
        except:
            print('Something wrong with finding cuts!')
            return

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'

        wcs_path = f'{file_path}/wcs/ref/corrected.fits'
        if not os.path.exists(wcs_path):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        with fits.open(wcs_path) as f:
            wcsItem = WCS(f[1].header)

        ra_center, dec_center = cutCentreCoords[cut-1]
        cut_corner = cutCorners[cut-1]
        # circumscribe the square cut (half its diagonal), not just inscribe it (half its
        # side), or objects crossing near the corners get silently missed
        radius_deg = cutSize * np.sqrt(2) * 21.0 / 3600.0

        for i in range(2 if part else 1):
            part_label = f' Part {i+1}' if part else ''
            cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            cutPath = f'{cutFolder}/{cutName}'
            base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
            times_path = f'{cutFolder}/{base}_Times.npy'

            if os.path.exists(cutPath):
                mjd = lk.read(cutPath).time.mjd
            elif os.path.exists(times_path):
                # raw cut TPF already cleaned up post-reduce() -- reduce()'s own saved frame
                # times cover the same thing (a strict subset if bad-quality frames were dropped)
                mjd = np.load(times_path)
            else:
                print(f'No cut or reduced times found to predict asteroids for '
                      f'(Cam {cam} Ccd {ccd} Cut {cut}{part_label})!')
                continue

            if self.verbose > 0:
                print(f'Predicting Asteroids for Cam {cam} CCD {ccd} Cut {cut} (of {n**2}){part_label}')

            try:
                # allow_download=False: this runs as a SLURM job on a compute node with no
                # internet access, so missing/uncovered kernel data must fail fast here rather
                # than have a worker process try (and fail) to reach MAST mid-integration
                ephemeris = predict_asteroids_for_footprint(
                    ra_center, dec_center, radius_deg,
                    mjd.min(), mjd.max(), mjd, wcsItem,
                    data_dir=f'{self.data_path}/mpc', allow_download=False)
            except RuntimeError as e:
                print(f'Cannot predict asteroids for Cam {cam} Ccd {ccd} Cut {cut}{part_label}: {e}')
                continue

            # store x,y local to the cut (matching the cut TPF's own pixel indexing),
            # not full-CCD, so results line up directly with the cut's reduced data
            if len(ephemeris):
                ephemeris['x'] -= cut_corner[0]
                ephemeris['y'] -= cut_corner[1]
                in_fov = ephemeris['x'].between(0,2*cutSize) & ephemeris['y'].between(0,2*cutSize)
            else:
                in_fov = ephemeris.index

            # every asteroid output except the asteroids.txt marker (checked directly in the
            # cut folder, matching cut.txt/reduced.txt's convention) lives in its own
            # subdirectory rather than cluttering the cut folder alongside everything else
            _Save_space(f'{cutFolder}/asteroids')
            save_table(ephemeris,f'{cutFolder}/asteroids/{base}_Asteroids.csv')
            plot_asteroid_trails(ephemeris[in_fov],f'{cutFolder}/asteroids/{base}_AsteroidTrails.png',footprint_size=2*cutSize)

            with open(f'{cutFolder}/asteroids.txt', 'w') as file:
                file.write('Predicted!')

            if self.verbose > 0:
                n_tracks = ephemeris['designation'].nunique() if len(ephemeris) else 0
                print(f'Cam {cam} CCD {ccd} Cut {cut}{part_label} asteroid prediction complete '
                      f'({n_tracks} tracks found).')
                print('\n')

    def _reduce_part_cuts(self,cam,ccd,n,cut,filepath):

        for i in range(2):
            cutFolder = f'{filepath}/Part{i+1}/Cut{cut}of{n**2}'
            cutName = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}.fits'
            cutPath = f'{cutFolder}/{cutName}'
            fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'
            if os.path.exists(fluxName):
                if self.verbose > 0:
                    print(f'Cam {cam} Chip {ccd} Cut {cut} Part {i+1} already reduced!')
            else:
                ts = t()
                if self.verbose > 0:
                    print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2}) Part {i+1} --')

                # -- Defining so can be deleted if failed -- #
                tessreduce = 0

                # -- reduce -- #
                tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,reduce=True,corr_correction=True,
                                            calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',
                                            prf_path='/fred/oz335/_local_TESS_PRFs',backend='multiprocessing',verbose=2)
                
                if self.verbose > 0:
                    print(f'--Reduction Part {i+1} Complete (Time: {((t()-ts)/60):.2f} mins)--')
                    print('\n')
                #tw = t()   # write timeStart
                
                # -- Saves information out as Numpy Arrays -- #
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Times.npy',tessreduce.lc[0])
                save_compact_array(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
                save_compact_array(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Background.npy',tessreduce.bkg)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Ref.npy',tessreduce.ref)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Mask.npy',tessreduce.mask)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Shifts.npy',tessreduce.shift)

                del (tessreduce)

                with open(f'{cutFolder}/reduced.txt', 'w') as file:
                    file.write(f'Reduced with TESSreduce version {tr.__version__}.')


    def reduce(self,cam,ccd,n,cut,part=False,injection=False,injection_dir='source_injection'):
        """
        Reduces a cut on a ccd using TESSreduce. bkg correlation 
        correction and final calibration are disabled due to time constraints.
        
        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 part cuts

        -------
        Creates
        -------
        Fits file in path with reduced data.

        """

        import tessreduce as tr
        from .localisation import CutWCS
        
        filepath = f'{self.path}/Cam{cam}/Ccd{ccd}'
        injection_dir = injection_dir if injection else '.'

        if part:
            self._reduce_part_cuts(cam,ccd,n,cut,filepath)
        else:
            cutFolder = f'{filepath}/Cut{cut}of{n**2}'
            cutBase = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}' 

            if injection:
                cutPath = None
                save_path = f'{cutFolder}/{injection_dir}/{cutBase}'
                flux = np.load(f'{save_path}_RawFlux.npy')
                mjd = np.load(f'{cutFolder}/{cutBase}_Times.npy')
                shifts = np.load(f'{cutFolder}/{cutBase}_Shifts.npy')
                wcs = CutWCS(self.data_path,self.sector,cam,ccd,cut,n)
            else:
                cutPath = f'{cutFolder}/{cutBase}.fits'
                save_path = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'
                flux = None
                mjd = None
                shifts = None
                wcs = None

            cut_corners,_,_,_ = self.find_cuts(cam,ccd,n,plot=False)

            with open(f'{self.path}/Cam{cam}/Ccd{ccd}/wcs/ref/reference.txt') as reffile:
                ref_ind = int(reffile.readlines()[0].split(': ')[-1])

            # fluxName = f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy'

            # if os.path.exists(fluxName):
            #     if self.verbose > 0:
            #         print(f'Cam {cam} Chip {ccd} cut {cut} already reduced!')
            # else:
            ts = t()
            if self.verbose > 0:
                print(f'--Reduction Cam {cam} Chip {ccd} Cut {cut} (of {n**2})--')
                


            # -- Defining so can be deleted if failed -- #
            tessreduce = 0

            # -- reduce -- #
            tessreduce = tr.tessreduce(tpf=cutPath,sector=self.sector,camera=cam,ccd=ccd,
                                       flux=flux,mjd=mjd,shifts=shifts,wcs=wcs,
                                        reduce=True,corr_correction=True,
                                        calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',col_offset=int(cut_corners[cut-1][0]),#-44,
                                        prf_path='/fred/oz335/_local_TESS_PRFs',vector_path='/fred/oz335/_local_TESS_vectors',
                                        ref_ind=ref_ind,quality_bitmask='hard',shift_method='sep_core',smooth_motion=False,
                                        orbit_ref=True,create_lc=False,timing=True,backend='multiprocessing',verbose=2)

            # # -- reduce -- #
            # tessreduce = tr.tessreduce(tpf=cutPath,flux=fluxPath,time=timePath,ref=refPath,
            #                            sector=self.sector,reduce=True,corr_correction=True,
            #                             calibrate=False,catalogue_path=f'{cutFolder}/local_gaia_cat.csv',col_offset=int(cut_corners[cut-1][0]),#-44,
            #                             prf_path='/fred/oz335/_local_TESS_PRFs',vector_path='/fred/oz335/_local_TESS_vectors',
            #                             ref_ind=ref_ind,quality_bitmask='hard',shift_method='sep_core',smooth_motion=False,
            #                             orbit_ref=True,create_lc=False,timing=True,backend='multiprocessing',verbose=2)
            
            if self.verbose > 0:
                print(f'--Reduction Complete (Time: {((t()-ts)/60):.2f} mins)--')
                print('\n')
            #tw = t()   # write timeStart
            
            # -- Saves information out as Numpy Arrays -- #
            np.save(f'{save_path}_Times.npy',tessreduce.mjd)
            np.save(f'{save_path}_ReducedFlux.npy',tessreduce.flux.astype(np.float32))
            np.save(f'{save_path}_Background.npy',tessreduce.bkg)
            np.save(f'{save_path}_Ref.npy',tessreduce.ref)
            np.save(f'{save_path}_Mask.npy',tessreduce.mask)
            np.save(f'{save_path}_Shifts.npy',tessreduce.shift)
            np.save(f'{save_path}_OrbitSegments.npy',tessreduce.orbit_segments)
            np.savez(f'{save_path}_OrbitRefs.npz',
                     **{str(k): v for k, v in tessreduce.orbit_refs.items()})

            del (tessreduce)

    def asteroid_lightcurves(self,cam,ccd,n,cut,part=False):
        """
        Forced aperture + PSF photometry, pixel-phase detrending, star-
        contamination flagging, and shift-and-stack for every asteroid
        predict_asteroids() found crossing this cut (see
        asteroid_photometry.py) -- runs after reduce(), since it needs the
        reduced flux cube; predict_asteroids()'s own frame indexing (from
        the raw, unreduced cut TPF) is re-derived against the REDUCED
        cube's own frame list first, since reduction can drop bad-quality
        frames.

        ------
        Inputs
        ------
        cam : int
            desired camera
        ccd : int
            desired ccd
        n : int
            n**2 cuts have been made
        cut : int
            specific cut

        ------
        Creates
        ------
        Per-cut forced aperture/PSF photometry, stacking summary, and
        stacked-lightcurve tables in the cut's folder.

        """

        import pandas as pd
        from astropy.io import fits
        from astropy.wcs import WCS
        from .asteroid_photometry import (forced_psf_photometry,
                                            match_ephemeris_to_reduced_frames, detrend_pixel_phase,
                                            local_gaia_cat_to_stars, flag_star_contamination,
                                            stack_lightcurves, STACK_SIG_TARGET, pool_offset_from_stacks)

        try:
            cutCorners, _, _, _ = self.find_cuts(cam=cam,ccd=ccd,n=n,plot=False,verbose=0)
        except:
            print('Something wrong with finding cuts!')
            return

        file_path = f'{self.path}/Cam{cam}/Ccd{ccd}'
        cut_corner = cutCorners[cut-1]

        wcs_path = f'{file_path}/wcs/ref/corrected.fits'
        if not os.path.exists(wcs_path):
            print('No corrected WCS found -- run fix_wcs first!')
            return
        with fits.open(wcs_path) as f:
            wcsItem = WCS(f[1].header)

        for i in range(2 if part else 1):
            part_label = f' Part {i+1}' if part else ''
            cutFolder = f'{file_path}/Part{i+1}/Cut{cut}of{n**2}' if part else f'{file_path}/Cut{cut}of{n**2}'
            base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}'

            asteroids_path = f'{cutFolder}/asteroids/{base}_Asteroids.csv'
            if not table_exists(asteroids_path):
                print(f'No asteroid predictions found for Cam {cam} Ccd {ccd} Cut {cut}{part_label} '
                      '-- run predict_asteroids first!')
                continue

            flux_path = f'{cutFolder}/{base}_ReducedFlux.npy'
            times_path = f'{cutFolder}/{base}_Times.npy'
            if not (os.path.exists(flux_path) and os.path.exists(times_path)):
                print(f'No reduced data found for Cam {cam} Ccd {ccd} Cut {cut}{part_label} -- run reduce first!')
                continue

            if self.verbose > 0:
                print(f'Building Asteroid Lightcurves for Cam {cam} CCD {ccd} Cut {cut} (of {n**2}){part_label}')

            ephemeris = load_table(asteroids_path)
            if len(ephemeris) == 0:
                print(f'No asteroids predicted for Cam {cam} Ccd {ccd} Cut {cut}{part_label}.')
                with open(f'{cutFolder}/asteroid_lightcurves.txt', 'w') as file:
                    file.write('No asteroids to process.')
                continue

            cube = np.load(flux_path)
            reduced_mjd = np.load(times_path)
            ephemeris = match_ephemeris_to_reduced_frames(ephemeris, reduced_mjd)

            gaia_cat_path = f'{cutFolder}/local_gaia_cat.csv'
            if os.path.exists(gaia_cat_path):
                gaia_cat = pd.read_csv(gaia_cat_path)
                stars = local_gaia_cat_to_stars(gaia_cat, wcsItem, cut_corner[0], cut_corner[1])
            else:
                stars = pd.DataFrame(columns=['x','y','mag'])

            # this cut's own AB zeropoint, if calibrate() has already run for it -- lets
            # pool_offset_from_stacks convert each track's fitted flux to a real magnitude
            # for a direct sanity check against the ephemeris's own predicted mag_expected.
            # None (uncalibrated) is a normal, handled case, not an error.
            zp_path = f'{cutFolder}/calibration/psf_calibration_zp.csv'
            zp_ab = None
            if os.path.exists(zp_path):
                try:
                    zp_ab = float(pd.read_csv(zp_path)['zp_ab'].iloc[0])
                except Exception:
                    zp_ab = None

            # Step 1: find the predicted-vs-measured position offset, straight off the raw
            # predicted ephemeris -- measure_stack_centroid_offset works from raw cube stamps
            # and its own joint position+flux fit's significance (not a forced-photometry
            # SNR), so no photometry needs to run before the offset is known.
            offset_x, offset_y, n_offset_tracks, offset_diagnostics = pool_offset_from_stacks(
                ephemeris, cube, self.sector, cam, ccd, cut_corner[0], cut_corner[1], zp_ab=zp_ab)

            if np.isfinite(offset_x) and np.isfinite(offset_y):
                # both the aperture and the (non-centroiding, fixed-position) PSF fit assume
                # the given x,y IS the source, so a real, measured predicted-vs-actual offset
                # left uncorrected here silently loses flux (an off-centre aperture misses
                # part of the source; an offset PSF template correlates less well with the
                # data and its best-fit amplitude comes out low) on every single measurement
                photometry_ephemeris = ephemeris.copy()
                photometry_ephemeris['x'] += offset_x
                photometry_ephemeris['y'] += offset_y
            else:
                # too few high-SNR tracks to trust a measured offset (pool_offset_from_stacks'
                # own min_tracks fallback) -- save at the uncorrected position instead
                photometry_ephemeris = ephemeris

            # Step 2: forced photometry, exactly once, at the position from step 1.
            # Aperture photometry is currently disabled -- not part of the pipeline for now
            # (forced_aperture_photometry itself is untouched in asteroid_photometry.py, just
            # not called here).
            psf_df = forced_psf_photometry(cube, photometry_ephemeris, self.sector, cam, ccd,
                                             cut_corner[0], cut_corner[1])
            psf_df = detrend_pixel_phase(psf_df)

            # Step 3: contamination flags, on the one photometry result actually being saved
            psf_df = flag_star_contamination(psf_df, stars, flux_col='flux_detrended')
            stack_summary, stacked_df = stack_lightcurves(psf_df)

            offset_df = pd.DataFrame([{'offset_x': offset_x, 'offset_y': offset_y,
                                        'n_tracks_used': n_offset_tracks}])

            save_table(psf_df,f'{cutFolder}/asteroids/{base}_AsteroidPSFPhotometry.csv')
            save_table(stack_summary.reset_index(),f'{cutFolder}/asteroids/{base}_AsteroidStackSummary.csv')
            save_table(stacked_df,f'{cutFolder}/asteroids/{base}_AsteroidStackedPhotometry.csv')
            save_table(offset_df,f'{cutFolder}/asteroids/{base}_AsteroidCutOffset.csv')
            save_table(offset_diagnostics,f'{cutFolder}/asteroids/{base}_AsteroidOffsetDiagnostics.csv')

            with open(f'{cutFolder}/asteroid_lightcurves.txt', 'w') as file:
                file.write('Done!')

            if self.verbose > 0:
                n_tracks = ephemeris['designation'].nunique()
                n_robust = int(((~stack_summary['stacking_needed']) |
                                 (stack_summary['achieved_sig'] >= STACK_SIG_TARGET)).sum())
                print(f'Cam {cam} CCD {ccd} Cut {cut}{part_label} asteroid lightcurves complete '
                      f'({n_tracks} tracks, {n_robust} robustly detected).')
                print('\n')