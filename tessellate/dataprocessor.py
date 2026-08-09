import os
import numpy as np
from time import time as t

from .tools import _Save_space, _Remove_emptys, _Extract_fits, _Print_buff
    
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
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_ReducedFlux.npy',tessreduce.flux)
                np.save(f'{cutFolder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{n**2}_Background.npy',tessreduce.bkg)
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