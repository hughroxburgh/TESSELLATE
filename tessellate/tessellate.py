from time import time as t
# ts = t()
from time import sleep

from glob import glob
import os
import re
import sys

VENV_PATH = sys.prefix

import numpy as np
# print(f'Imported easy functions ({ts-t():.0f}s)')

# ts = t()
from .tools import delete_files, _Print_buff, _Save_space, _Check_job_status
# print(f'Imported .tools functions ({ts-t():.0f}s)')

class Tessellate():
    """
    Parent Class for Tessellation runs.
    """

    def __init__(self,data_path,sector=None,cam=None,ccd=None,n=None,
                 verbose=2,ask_config=True,save_config=True,
                 job_output_path=None,working_path=None,injection_dir='source_injection',
                 download_number=None,cube_time=None,cube_mem=None,cube_cpu=None,
                 cuts=None,cut_time=None,cut_mem=None,cut_cpu=None,
                 predict_asteroids_time=None,predict_asteroids_cpu=None,predict_asteroids_mem=None,
                 reduce_time=None,reduce_cpu=None,reduce_mem=None,
                 asteroid_lightcurves_time=None,asteroid_lightcurves_cpu=None,asteroid_lightcurves_mem=None,
                 calibrate_time=None,calibrate_cpu=None,calibrate_mem=None,
                 search_time=None,search_cpu=None,search_mem=None,detect_mode='both',time_bins=None,
                 plot_time=None,plot_cpu=None,plot_mem=None,
                 download=None,make_cube=None,fix_wcs=None,make_cuts=None,predict_asteroids=None,
                 reduce=None,asteroid_lightcurves=None,calibrate=None,search=None,plot=None,delete=None,
                 injection=False,overwrite=None,reset_logs=None,use_suggestions=False,
                 go=True):
        
        """
        Initialise.
        
        ------
        Inputs
        ------
        data_path : str 
            /path/to/data/location (use '/fred/oz335/TESSdata')
        job_output_path : str
            /path/to/job_log/output/location
        working_path : str
            /path/to/working_directory
        
        The Remaining INPUTS/OPTIONS/PROPERTIES can == 'None', will require input upon initialisation.

        sector : int
            TESS sector to operate upon
        cam : int, 'all'
            desired cam(s)
        ccd : int, 'all'
            desired ccd(s)

        ------
        Options
        ------

        n : int
            n^2 number of cuts will be made
        verbose : int
            0 = nothing, 1 = some output, 2 = full output
        download_number : 'all', int
            number of FFIs to download and operate on
        cuts : 'all', int
            specific cuts to operate upon per cam/ccd

        download : bool
            if True, download FFIs
        make_cube : bool
            if True, make cube
        make_cuts : bool
            if True, make cuts
        predict_asteroids : bool
            if True, predict catalogued asteroids crossing each cut's footprint
        reduce : bool
            if True, reduce cuts
        asteroid_lightcurves : bool
            if True, build forced photometry lightcurves for predicted asteroids
        search : bool
            if True, run search pipeline
        delete : bool
            if True, delete FFIs after process
        
        ------
        Run Properties
        ------

        cube_time : str
            time allowed for cube generation, form = 'hh:mm:ss'
        cube_cpu : int
            number of cpu to request for cube generation
        cube_mem : int
            memory/cpu for cube generation (in GB)

        cut_time : str
            time allowed for cut generation, form = 'hh:mm:ss'
        cut_cpu : int
            number of cpu to request for cut generation
        cut_mem : int
            memory/cpu for cut generation (in GB)

        predict_asteroids_time : str
            time allowed for asteroid prediction, form = 'hh:mm:ss'
        predict_asteroids_cpu : int
            number of cpu to request for asteroid prediction
        predict_asteroids_mem : int
            memory/cpu for asteroid prediction (in GB)

        reduce_time : str
            time allowed for cut reduction, form = 'hh:mm:ss'
        reduce_cpu : int
            number of cpu to request for cut reduction
        reduce_mem : int
            memory/cpu for cut reduction (in GB)

        asteroid_lightcurves_time : str
            time allowed for asteroid lightcurve generation, form = 'hh:mm:ss'
        asteroid_lightcurves_cpu : int
            number of cpu to request for asteroid lightcurve generation
        asteroid_lightcurves_mem : int
            memory/cpu for asteroid lightcurve generation (in GB)

        search_time : str
            time allowed for transient event search, form = 'hh:mm:ss'
        search_cpu : int
            number of cpu to request for transient event search
        search_mem : int
            memory/cpu for transient event search (in GB)

        """
        
        if (job_output_path is None) | (working_path is None):
            m = 'Ensure you specify paths for job output and working path!\n'
            raise ValueError(m)
        else:
            self.job_output_path = job_output_path
            self.working_path = working_path

        self.sector = sector
        self.data_path = data_path

        self.cam = cam
        self.ccd = ccd
        self.verbose=verbose

        self.download_number = download_number
        self.n = n
        self.cuts = cuts

        self.cube_time = cube_time
        self.cube_mem = cube_mem
        self.cube_cpu = cube_cpu

        self.cut_time = cut_time
        self.cut_mem = cut_mem
        self.cut_cpu = cut_cpu

        self.predict_asteroids_time = predict_asteroids_time
        self.predict_asteroids_mem = predict_asteroids_mem
        self.predict_asteroids_cpu = predict_asteroids_cpu

        self.injection = injection
        self._inj_dir = injection_dir if injection else '.'

        self.reduce_time = reduce_time
        self.reduce_cpu = reduce_cpu
        self.reduce_mem = reduce_mem

        self.asteroid_lightcurves_time = asteroid_lightcurves_time
        self.asteroid_lightcurves_cpu = asteroid_lightcurves_cpu
        self.asteroid_lightcurves_mem = asteroid_lightcurves_mem

        self.calibrate_time = calibrate_time
        self.calibrate_cpu = calibrate_cpu
        self.calibrate_mem = calibrate_mem

        self.search_time = search_time
        self.search_mem = search_mem
        self.search_cpu = search_cpu
        self.detect_mode = detect_mode
        self.time_bins = time_bins

        self.plot_time = plot_time
        self.plot_mem = plot_mem
        self.plot_cpu = plot_cpu

        self.ask_config = ask_config
        self.skip = []

        # -- Allows for no actual initialisation (TessTransient) -- #
        if go:
            self.run_tessellate(download,make_cube,fix_wcs,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot,delete,
                                overwrite,reset_logs,save_config,use_suggestions)

    def run_tessellate(self,download,make_cube,fix_wcs,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot,delete,
                       overwrite,reset_logs,save_config,use_suggestions):

        # -- Initialise and check for previous config file -- #
        load_prev = self._initialise()

        if load_prev:

            # -- Load config -- #
            download, make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot, delete = self._load_config()

        else:
            # -- Confirm Run Properties -- #
            self._run_properties()

            # -- Get time/cpu/memory suggestions depending on sector -- #
            suggestions = self._sector_suggestions()

            # -- Ask for which tessellation steps to perform -- #
            download, make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot, delete = self._which_processes(download, make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot, delete)

            # -- Ask for inputs -- #
            if download:
                self._download_properties()

            if make_cube:
                self._cube_properties(suggestions[0],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_cubing_logs')

            if make_cuts:
                self._cut_properties(suggestions[1],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_cutting_logs')

            if predict_asteroids:
                self._predict_asteroids_properties(make_cuts,suggestions[6],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_asteroid_prediction_logs')

            if reduce:
                self._reduce_properties(make_cuts,suggestions[2],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_reduction_logs')

            if asteroid_lightcurves:
                self._asteroid_lightcurves_properties(make_cuts or predict_asteroids,suggestions[7],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_asteroid_lightcurves_logs')

            if calibrate:
                self._calibrate_properties(reduce, suggestions[5],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_calibration_logs')

            if search:
                cutting_reducing = make_cuts | reduce
                self._search_properties(cutting_reducing,suggestions[3],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_search_logs')

            if plot:
                self._plotting_properties(search,suggestions[4],use_suggestions)
                _Save_space(f'{self.job_output_path}/tessellate_plotting_logs')

            if save_config:
                self._write_config(download,make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot, delete)

        # -- Check for overwriting -- #
        if overwrite != False:
            self._overwrite_suggestions(make_cube, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot)
        else:
            self.overwrite = None

        # -- Reset Job Logs -- #
        if reset_logs != False:
            self._reset_logs(make_cube,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot)

        # -- Run Processes -- #
        if download:
            self.download()

        if make_cube:
            make_cube = self.make_cube()

        if fix_wcs:
            self.fix_wcs(cube_status=make_cube)

        if make_cuts:
            self.make_cuts()

        if predict_asteroids:
            predict_asteroids = self.predict_asteroids()   # returns prediction slurm job ids for use in asteroid_lightcurves

        if reduce:
            reduce = self.reduce()     # returns reduction slurm job ids for use in transient search

        if asteroid_lightcurves:
            self.asteroid_lightcurves(prediction_status=predict_asteroids)

        if calibrate:
            self.calibrate(reduction_status=reduce)
            reduce = False

        if search:
            self.transient_search(calibrating=calibrate,reduction_status=reduce)
        
        if plot:
            self.transient_plot(searching=search)

        if delete:
            delete_files(filetype='ffis',data_path=self.data_path,sector=self.sector,part=False)  

    def _initialise(self):

        print('\n')
        print(_Print_buff(50,f'Initialising Tessellation Run'))

        load_prev = False
        if self.ask_config:
            file = f'{self.working_path}/tessellate_config.txt'
            if os.path.exists(file):
                load = input(f'Load from previous config ({file})? [y/n] = ') 
                done = False
                while not done:
                    if load.lower() == 'y':
                        load_prev = True
                        done=True
                    elif load.lower() == 'n':
                        load_prev = False
                        done=True
                    else:
                        load = input(f'  Invalid choice! Load from previous config ({file})? [y/n] = ')
                
                if load_prev:
                    print('Loading config')

        print('\n')

        return load_prev
    
    def _load_config(self):

        import configparser
        import ast
        import sys

        config_file = f'{self.working_path}/tessellate_config.txt'

        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve case of keys
        parser.read(config_file)
        
        # Helper to evaluate Python literals safely
        def parse_value(val):
            val = val.strip()
            try:
                # safely evaluate strings like True, False, numbers, lists, tuples
                return ast.literal_eval(val)
            except Exception:
                # fallback to string
                return val.strip("'").strip('"')
        
        # Base section
        self.sector = parse_value(parser['base'].get('sector'))
        self.cam = parse_value(parser['base'].get('cam'))
        self.ccd = parse_value(parser['base'].get('ccd'))
        
        suggestions = self._sector_suggestions()

        print(f'   - Sector = {self.sector}')
        print(f'   - Cam = {self.cam}')
        print(f'   - CCD = {self.ccd}')
        print('\n')
        
        # Download
        if 'download' in parser:
            download = parse_value(parser['download'].get('download', False))
            self.download_number = parse_value(parser['download'].get('download_number', None))  
            print(f'   - Download FFIs? [y/n] = y')
        else:
            download = False
            print(f'   - Download FFIs? [y/n] = n')
        
        # Make cubes
        if 'make_cubes' in parser:
            make_cube = parse_value(parser['make_cubes'].get('make_cubes', False))
            self.cube_time = parse_value(parser['make_cubes'].get('cube job time', None))
            self.cube_mem = parse_value(parser['make_cubes'].get('cube job memory', None))
            self.cube_cpu = parse_value(parser['make_cubes'].get('cube job cpu', None))
            print(f'   - Make Cube(s)? [y/n] = y')
        else:
            make_cube = False
            print(f'   - Make Cube(s)? [y/n] = n')

        # Fix WCS 
        if 'fix_wcs' in parser:
            fix_wcs = parse_value(parser['fix_wcs'].get('fix_wcs', False))
        else:
            fix_wcs = False
            print(f'   - Fix WCS? [y/n] = n')
        
        # Cut properties
        if 'cut_properties' in parser:
            self.n = parse_value(parser['cut_properties'].get('n', None))
            self.cuts = parse_value(parser['cut_properties'].get('cuts', None))
        
        # Make cuts
        if 'make_cuts' in parser:
            make_cuts = parse_value(parser['make_cuts'].get('make_cuts', False))
            self.cut_time = parse_value(parser['make_cuts'].get('cut job time', None))
            self.cut_mem = parse_value(parser['make_cuts'].get('cut job memory', None))
            self.cut_cpu = parse_value(parser['make_cuts'].get('cut job cpu', None))
            print(f'   - Make Cut(s)? [y/n] = y')
        else:
            make_cuts = False
            print(f'   - Make Cut(s)? [y/n] = n')

        # Predict Asteroids
        if 'predict_asteroids' in parser:
            predict_asteroids = parse_value(parser['predict_asteroids'].get('predict_asteroids', False))
            self.predict_asteroids_time = parse_value(parser['predict_asteroids'].get('predict_asteroids job time', None))
            self.predict_asteroids_mem = parse_value(parser['predict_asteroids'].get('predict_asteroids job memory', None))
            self.predict_asteroids_cpu = parse_value(parser['predict_asteroids'].get('predict_asteroids job cpu', None))
            print(f'   - Predict Asteroids on Cut(s)? [y/n] = y')
        else:
            predict_asteroids = False
            print(f'   - Predict Asteroids on Cut(s)? [y/n] = n')

        # Reduce
        if 'reduce' in parser:
            reduce = parse_value(parser['reduce'].get('reduce', False))
            self.reduce_time = parse_value(parser['reduce'].get('reduce job time', None))
            self.reduce_mem = parse_value(parser['reduce'].get('reduce job memory', None))
            self.reduce_cpu = parse_value(parser['reduce'].get('reduce job cpu', None))
            print(f'   - Reduce Cut(s)? [y/n] = y')
        else:
            reduce = False
            print(f'   - Reduce Cut(s)? [y/n] = n')

        # Asteroid Lightcurves
        if 'asteroid_lightcurves' in parser:
            asteroid_lightcurves = parse_value(parser['asteroid_lightcurves'].get('asteroid_lightcurves', False))
            self.asteroid_lightcurves_time = parse_value(parser['asteroid_lightcurves'].get('asteroid_lightcurves job time', None))
            self.asteroid_lightcurves_mem = parse_value(parser['asteroid_lightcurves'].get('asteroid_lightcurves job memory', None))
            self.asteroid_lightcurves_cpu = parse_value(parser['asteroid_lightcurves'].get('asteroid_lightcurves job cpu', None))
            print(f'   - Build Asteroid Lightcurves on Cut(s)? [y/n] = y')
        else:
            asteroid_lightcurves = False
            print(f'   - Build Asteroid Lightcurves on Cut(s)? [y/n] = n')

        # Calibrate
        if 'calibrate' in parser:
            calibrate = parse_value(parser['calibrate'].get('calibrate', False))
            self.calibrate_time = parse_value(parser['calibrate'].get('calibrate job time', None))
            self.calibrate_mem = parse_value(parser['calibrate'].get('calibrate job memory', None))
            self.calibrate_cpu = parse_value(parser['calibrate'].get('calibrate job cpu', None))
            print(f'   - Flux Calibrate Cut(s)? [y/n] = y')
        else:
            calibrate = False
            print(f'   - Flux Calibrate Cut(s)? [y/n] = n')

        # Search
        if 'search' in parser:
            search = parse_value(parser['search'].get('search', False))
            self.search_time = parse_value(parser['search'].get('search job time', None))
            self.search_mem = parse_value(parser['search'].get('search job memory', None))
            self.search_cpu = parse_value(parser['search'].get('search job cpu', None))
            self.detect_mode = parse_value(parser['search'].get('search detection mode', None))
            self.time_bins = parse_value(parser['search'].get('search time bins', None))
            print(f'   - Run Transient Search on Cut(s)? [y/n] = y')
        else:
            search = False
            print(f'   - Run Transient Search on Cut(s)? [y/n] = n')
        
        # Plot
        if 'plot' in parser:
            plot = parse_value(parser['plot'].get('plot', False))
            self.plot_time = parse_value(parser['plot'].get('plot job time', None))
            self.plot_mem = parse_value(parser['plot'].get('plot job memory', None))
            self.plot_cpu = parse_value(parser['plot'].get('plot job cpu', None))
            print(f'   - Run Transient Plotting on Cut(s)? [y/n] = y')
        else:
            plot = False
            print(f'   - Run Transient Plotting on Cut(s)? [y/n] = n')
        
        # Options
        if 'options' in parser:
            self.verbose = parse_value(parser['options'].get('verbose', False))
            delete = parse_value(parser['options'].get('delete ffis', False))
            if delete:
                print(f'   - Delete all FFIs upon completion? [y/n] = y')
            else:
                print(f'   - Delete all FFIs upon completion? [y/n] = n')

        print('\n')

        if calibrate:
            print(f"   - Calibrate Batch Time ['h:mm:ss'] = {self.calibrate_time}")
            print(f"   - Calibrate Mem/CPU = {self.calibrate_mem}")
            print(f"   - Calibrate Num CPUs Needed = {self.calibrate_cpu}")
            print('\n')

        if download:
            print(f'   - Download Number [int,all] = {self.download_number}')
            print('\n')

        if make_cube:
            print(f"   - Cube Batch Time ['h:mm:ss'] = {self.cube_time}")
            print(f"   - Cube Mem/CPU = {self.cube_mem}")
            print(f"   - Cube Num CPUs Needed = {self.cube_cpu}")
            print('\n')

        if make_cuts | reduce | calibrate | search | plot:
            print(f"   - n (Number of Cuts = n^2) = {self.n}")
    
            if self.cuts == list(range(1,17)):
                cut_printout = 'all'
            else:
                cut_printout = self.cuts
            print(f"   - Cut [1-16,all] = {cut_printout}")
            print('\n')

        if make_cuts:
            print(f"   - Cut Batch Time ['h:mm:ss'] = {self.cut_time}")
            print(f"   - Cut Mem/CPU = {self.cut_mem}")
            print(f"   - Cut Num CPUs Needed = {self.cut_cpu}")
            print('\n')

        if predict_asteroids:
            print(f"   - Predict Asteroids Batch Time ['h:mm:ss'] = {self.predict_asteroids_time}")
            print(f"   - Predict Asteroids Num CPUs = {self.predict_asteroids_cpu}")
            print(f"   - Predict Asteroids Mem/CPU = {self.predict_asteroids_mem}")
            print('\n')

        if reduce:
            print(f"   - Reduce Batch Time ['h:mm:ss'] = {self.reduce_time}")
            print(f"   - Reduce Mem/CPU = {self.reduce_mem}")
            print(f"   - Reduce Num CPUs Needed = {self.reduce_cpu}")
            print('\n')

        if asteroid_lightcurves:
            print(f"   - Asteroid Lightcurves Batch Time ['h:mm:ss'] = {self.asteroid_lightcurves_time}")
            print(f"   - Asteroid Lightcurves Num CPUs = {self.asteroid_lightcurves_cpu}")
            print(f"   - Asteroid Lightcurves Mem/CPU = {self.asteroid_lightcurves_mem}")
            print('\n')

        if search:
            print(f"   - Search Batch Time ['h:mm:ss'] = {self.search_time}")
            print(f"   - Search Mem/CPU = {self.search_mem}")
            print(f"   - Search Num CPUs Needed = {self.search_cpu}")
            print(f"   - Search Time Bins = {self.time_bins}")
            print('\n')

        if plot:
            print(f"   - Plotting Batch Time ['h:mm:ss'] = {self.plot_time}")
            print(f"   - Plotting Mem/CPU = {self.plot_mem}")
            print(f"   - Plotting Num CPUs Needed = {self.plot_cpu}")
            print('\n')

        go = input('   Ready? [y/n] = ')
        done = False
        while not done:
            if go.lower() == 'y':
                go = True
                done=True
            elif go.lower() == 'n':
                srs = input('      Cancel? [y/n] = ')
                innerdone = False
                while not innerdone:
                    if srs.lower() == 'y':
                        print('\n')
                        sys.exit(0)
                    elif srs.lower() != 'n':
                        srs = input('         Invalid choice! Cancel? [y/n] = ')
                    else:
                        innerdone = True
                go = input('   Ready? [y/n] = ')
            else:
                go = input('      Invalid choice! Ready? [y/n] = ')


        return download,make_cube,fix_wcs,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot,delete

    def _sector_suggestions(self):
        """
        Generate suggestions for slurm job runtime, cpu allocation, memory based on sector (tested but temperamental)
        """

        primary_mission = range(1,27)       # ~1200 FFIs , 30 min cadence
        secondary_mission = range(27,56)    # ~3600 FFIs , 10 min cadence
        tertiary_mission = range(56,100)    # ~12000 FFIs , 200 sec cadence

        self.part = False
        if self.sector in primary_mission:
            cube_time_sug = '45:00'
            cube_mem_sug = '20G'
            cube_mem_req = 60

            cut_time_sug = '10:00'
            cut_mem_sug = '20G'
            cut_mem_req = 20

            # scaled from the secondary tier's real benchmark by frame count -- ~1200 vs ~3600
            # FFIs/sector (30min vs 10min cadence, same ~27-day window) is the actual driver of
            # both stages' runtime (precise_ephemeris's per-frame loop; one PSF fit per frame
            # per track), a ~0.33x ratio -- not itself benchmarked on a primary-mission sector
            predict_asteroids_time_sug = '5:00'
            predict_asteroids_cpu_sug = '8'
            predict_asteroids_mem_req = 24

            reduce_time_sug = '30:00'
            reduce_cpu_sug = '8'
            reduce_mem_req = 64

            # forced aperture/PSF photometry is now parallelised across tracks (shared-memory
            # cube, see asteroid_photometry.py's _run_parallel_by_track), so cpu is worth
            # scaling again -- time halved from the previous single-threaded suggestion as a
            # conservative estimate of the real speedup, not yet directly re-benchmarked in
            # parallel; mem bumped for each worker's own PRF cache (bounded, but duplicated
            # per worker) on top of the shared (not duplicated) cube
            asteroid_lightcurves_time_sug = '8:00'
            asteroid_lightcurves_cpu_sug = '8'
            asteroid_lightcurves_mem_req = 16

            calibrate_time_sug = '10:00'
            calibrate_cpu_sug = '8'
            calibrate_mem_req = 8

            search_time_sug = '20:00'
            search_cpu_sug = '32'
            search_mem_req = 32
            search_time_bins = '30min,2hr,12hr'

            plot_time_sug = '10:00'
            plot_cpu_sug = '32'
            plot_mem_req = 64

        elif self.sector in secondary_mission:
            cube_time_sug = '1:45:00'
            cube_mem_sug = '20G'
            cube_mem_req = 140

            # cut_time_sug = '2:00:00'
            # cut_mem_sug = '20G'
            # cut_mem_req = 60

            cut_time_sug = '15:00'
            cut_mem_sug = '20G'
            cut_mem_req = 60

            # real-benchmarked (Sector 32, the only tier actually tested): max observed
            # 3 min / 3.6GB peak (Cam1 Ccd1 Cut57, 54 crossing tracks, 215 stage-2
            # survivors -- a genuinely dense, near-ecliptic cut) with cpu_sug=8 at ~70%
            # average utilisation, vs. the previous unbenchmarked 30:00/128G guess
            predict_asteroids_time_sug = '10:00'
            predict_asteroids_cpu_sug = '8'
            predict_asteroids_mem_req = 24

            # reduce_time_sug = '1:15:00'
            # reduce_cpu_sug = '32'
            # reduce_mem_req = 160

            reduce_time_sug = '45:00'
            reduce_cpu_sug = '8'
            reduce_mem_req = 64

            # revised after a real Sector 32 full-run: the 2-cut benchmark (15:00/8G) undershot
            # badly on the sector's denser CCDs -- OOM-killed jobs peaked (by periodic sampling)
            # at only 3.8/8GB, meaning real transient spikes exceed what the sampling caught, and
            # TIMEOUTs hit cuts with far more tracks than Cut57's 54 (e.g. Cam1 Ccd3, a much
            # lower-ecliptic-latitude CCD). Widened with real margin rather than re-trimming.
            # Now also parallelised across tracks -- cpu scaled back up (see primary tier),
            # time halved as a conservative not-yet-benchmarked-in-parallel estimate, and
            # _cut_asteroid_lightcurves additionally scales both per-cut by real track count
            asteroid_lightcurves_time_sug = '15:00'
            asteroid_lightcurves_cpu_sug = '8'
            asteroid_lightcurves_mem_req = 28

            calibrate_time_sug = '10:00'
            calibrate_cpu_sug = '8'
            calibrate_mem_req = 8

            # search_time_sug = '30:00'
            # search_cpu_sug = '32'
            # search_mem_req = 64
            # search_time_bins = '10min'
            search_time_sug = '45:00'
            search_cpu_sug = '32'
            search_mem_req = 32
            search_time_bins = '10min,30min,2hr,12hr'

            plot_time_sug = '10:00'
            plot_cpu_sug = '32'
            plot_mem_req = 64

        elif self.sector in tertiary_mission:
            self.part = True

            cube_time_sug = '6:00:00'
            cube_mem_sug = '20G'
            cube_mem_req = 200

            cut_time_sug = '3:00:00'
            cut_mem_sug = '20G'
            cut_mem_req = 100

            # not itself benchmarked on a tertiary-mission sector. Scaled from the secondary
            # tier's real 3min/7:11 worst case by frame count -- ~12000 vs ~3600 FFIs/sector
            # (200sec vs 10min cadence, same ~27-day window), a ~3.3x ratio, the actual driver
            # of both stages' runtime (precise_ephemeris's per-frame loop; one PSF fit per
            # frame per track)
            predict_asteroids_time_sug = '30:00'
            predict_asteroids_cpu_sug = '16'
            predict_asteroids_mem_req = 72

            reduce_time_sug = '3:00:00'
            reduce_cpu_sug = '32'
            reduce_mem_req = 200

            # scaled from the secondary tier's post-full-run-revision values (see there for why)
            asteroid_lightcurves_time_sug = '45:00'
            asteroid_lightcurves_cpu_sug = '16'
            asteroid_lightcurves_mem_req = 64

            calibrate_time_sug = '10:00'
            calibrate_cpu_sug = '8'
            calibrate_mem_req = 8

            search_time_sug = '1:00:00'
            search_cpu_sug = '32'
            search_mem_req = 60
            search_time_bins = '200sec'

            plot_time_sug = '10:00'
            plot_cpu_sug = '32'
            plot_mem_req = 50

        suggestions = [[cube_time_sug,cube_mem_sug,cube_mem_req],
                       [cut_time_sug,cut_mem_sug,cut_mem_req],
                       [reduce_time_sug,reduce_cpu_sug,reduce_mem_req],
                       [search_time_sug,search_cpu_sug,search_mem_req,search_time_bins],
                       [plot_time_sug,plot_cpu_sug,plot_mem_req],
                       [calibrate_time_sug,calibrate_cpu_sug,calibrate_mem_req],
                       [predict_asteroids_time_sug,predict_asteroids_cpu_sug,predict_asteroids_mem_req],
                       [asteroid_lightcurves_time_sug,asteroid_lightcurves_cpu_sug,asteroid_lightcurves_mem_req]]

        return suggestions

    def _run_properties(self):
        """
        Confirm sector, cam, ccd properties.
        """

        if self.sector is None:
            sector = input('   - Sector = ')
            done = False
            while not done:
                try:
                    sector = int(sector)
                    if 0 < sector < 100:
                        self.sector = sector
                        done = True
                    else:
                        sector = input('      Invalid choice! Sector = ')
                except:
                    sector = input('      Invalid choice! Sector = ')
        elif 0 < self.sector < 100:
            print(f'   - Sector = {self.sector}')
        else:
            e = f'Invalid Sector Input of {self.sector}\n'
            raise ValueError(e)


        if self.cam is None:
            cam = input('   - Cam [1,2,3,4,all] = ')
            done = False
            while not done:
                if cam in ['1','2','3','4']:
                    self.cam = [int(cam)]
                    done = True
                elif cam.lower() == 'all':
                    self.cam = [1,2,3,4]
                    done = True
                elif cam == '34':
                    self.cam = [3,4]
                    done = True
                else:
                    cam = input('      Invalid choice! Cam [1,2,3,4,all] = ')
        elif self.cam == 'all':
            print(f'   - Cam = all')
            self.cam = [1,2,3,4]
        elif self.cam in [1,2,3,4]:
            print(f'   - Cam = {self.cam}')
            self.cam = [self.cam]
        else:
            e = f'Invalid Camera Input of {self.cam}\n'
            raise ValueError(e)
        

        if self.ccd is None:
            ccd = input('   - CCD [1,2,3,4,all] = ')
            done = False
            while not done:
                if ccd in ['1','2','3','4']:
                    self.ccd = [int(ccd)]
                    done = True
                elif ccd.lower() == 'all':
                    self.ccd = [1,2,3,4]
                    done = True
                else:
                    ccd = input('      Invalid choice! CCD [1,2,3,4,all] = ')
        elif self.ccd == 'all':
            print(f'   - CCD = all')
            self.ccd = [1,2,3,4]
        elif self.ccd in [1,2,3,4]:
            print(f'   - CCD = {self.ccd}')
            self.ccd = [self.ccd]
        else:
            e = f'Invalid CCD Input of {self.ccd}\n'
            raise ValueError(e)
        
        print('\n')
    
    def _which_processes(self,download,make_cube,fix_wcs,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot,delete):

        if download is None:
            d = input('   - Download FFIs? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    download = True
                    done=True
                elif d.lower() == 'n':
                    download = False
                    done=True
                else:
                    d = input('      Invalid choice! Download FFIs? [y/n] = ')

        if make_cube is None:
            d = input('   - Make Cube(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    make_cube = True
                    done=True
                elif d.lower() == 'n':
                    make_cube = False
                    done=True
                else:
                    d = input('      Invalid choice! Make Cube(s)? [y/n] = ')

        if fix_wcs is None:
            d = input('   - Fix WCS? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    fix_wcs = True
                    done=True
                elif d.lower() == 'n':
                    fix_wcs = False
                    done=True
                else:
                    d = input('      Invalid choice! Fix WCS? [y/n] = ')

        if make_cuts is None:
            d = input('   - Make Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    make_cuts = True
                    done=True
                elif d.lower() == 'n':
                    make_cuts = False
                    done=True
                else:
                    d = input('      Invalid choice! Make Cut(s)? [y/n] = ')

        if predict_asteroids is None:
            d = input('   - Predict Asteroids on Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    predict_asteroids = True
                    done=True
                elif d.lower() == 'n':
                    predict_asteroids = False
                    done=True
                else:
                    d = input('      Invalid choice! Predict Asteroids on Cut(s)? [y/n] = ')

        if reduce is None:
            d = input('   - Reduce Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    reduce = True
                    done=True
                elif d.lower() == 'n':
                    reduce = False
                    done=True
                else:
                    d = input('      Invalid choice! Reduce Cut(s)? [y/n] = ')

        if asteroid_lightcurves is None:
            d = input('   - Build Asteroid Lightcurves on Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    asteroid_lightcurves = True
                    done=True
                elif d.lower() == 'n':
                    asteroid_lightcurves = False
                    done=True
                else:
                    d = input('      Invalid choice! Build Asteroid Lightcurves on Cut(s)? [y/n] = ')

        if calibrate is None:
            d = input('   - Flux Calibrate Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    calibrate = True
                    done = True
                elif d.lower() == 'n':
                    calibrate = False
                    done = True
                else:
                    d = input('      Invalid choice! Flux Calibrate Cut(s)? [y/n] = ')

        if search is None:
            d = input('   - Run Transient Search on Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    search = True
                    done=True
                elif d.lower() == 'n':
                    search = False
                    done=True
                else:
                    d = input('      Invalid choice! Run Transient Search on Cut(s)? [y/n] = ')
        
        if plot is None:
            d = input('   - Run Transient Plotting on Cut(s)? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    plot = True
                    done=True
                elif d.lower() == 'n':
                    plot = False
                    done=True
                else:
                    d = input('      Invalid choice! Run Transient Plotting on Cut(s)? [y/n] = ')            

        if delete is None:
            d = input('   - Delete all FFIs upon completion? [y/n] = ')
            done = False
            while not done:
                if d.lower() == 'y':
                    delete = True
                    done=True
                elif d.lower() == 'n':
                    delete = False
                    done=True
                else:
                    d = input('      Invalid choice! Delete all FFIs upon completion? [y/n] = ')

        print('\n')

        return download, make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot, delete
    
    def _download_properties(self):
        """
        Confirm download process properties.
        """

        if self.download_number is None:
            dNum = input("   - Download Number [int,all] = ")
            done = False
            while not done:
                try:
                    dNum = int(dNum)
                    if 0 < dNum < 14000:
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input("      Invalid choice! Download Number [int,all] = ")
                except:
                    if dNum == 'all':
                        self.download_number = dNum
                        done = True
                    else:
                        dNum = input("      Invalid choice! Download Number [int,all] = ")
        elif self.download_number == 'all':
            print(f'   - Download Number = all')
        elif 0 < self.download_number < 14000:
            print(f'   - Download Number = {self.download_number}')
        else:
            e = f'Invalid Download Number Input of {self.download_number}\n'
            raise ValueError(e)
        
        print('\n')
    
    def _cube_properties(self,suggestions,use_suggestions):
        """
        Confirm cube generation process properties.
        """

        if self.cube_time is None:
            if use_suggestions:
                self.cube_time = suggestions[0]
                print(f'   - Cube Batch Time = {self.cube_time}')
            else:
                cube_time = input(f"   - Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in cube_time:
                        self.cube_time = cube_time
                        done = True
                    else:
                        cube_time = input(f"      Invalid format! Cube Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Cube Batch Time = {self.cube_time}')

        
        if self.cube_mem is None:
            if use_suggestions:
                self.cube_mem = int(suggestions[1][:-1])
                print(f'   - Cube Mem/CPU = {self.cube_mem}G')
            else:
                cube_mem = input(f"   - Cube Mem/CPU ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try: 
                        cube_mem = int(cube_mem)
                        if 0<cube_mem < 500:
                            self.cube_mem = cube_mem
                            done=True
                        else:
                            cube_mem = input(f"      Invalid format! Cube Mem/CPU ({suggestions[1]} suggested) = ")
                    except:
                        if cube_mem[-1].lower() == 'g':
                            self.cube_mem = cube_mem[:-1]
                            done = True
                        else:
                            cube_mem = input(f"      Invalid format! Cube Mem/CPU ({suggestions[1]} suggested) = ")

        elif 0 < self.cube_mem < 500:
            print(f'   - Cube Mem/CPU = {self.cube_mem}G')
        else:
            e = f"Invalid Cube Mem/CPU Input of {self.cube_mem}\n"
            raise ValueError(e)
        
                
        if type(self.download_number) == int:
            cube_cpu = input(f"   - Cube Num CPUs [1-32] = ")
            done = False
            while not done:
                try:
                    cube_cpu = int(cube_cpu)
                    if 0 < cube_cpu < 33:
                        self.cube_cpu = cube_cpu
                        done = True
                    else:
                        cube_cpu = input(f"      Invalid format! Cube Num CPUs [1-32] = ")
                except:
                    cube_cpu = input(f"      Invalid format! Cube Num CPUs [1-32] = ")
        else:
            if suggestions[2] % int(self.cube_mem) != 0:
                self.cube_cpu = suggestions[2] // int(self.cube_mem) + 1
            else:
                self.cube_cpu = suggestions[2] // int(self.cube_mem)

            print(f'   - Cube Num CPUs Needed = {self.cube_cpu}')

        print('\n')

    def _cut_properties(self,suggestions,use_suggestions):
        """
        Confirm cut generation process properties.
        """

        if self.n is None:
            n = input('   - n (Number of Cuts = n^2) = ')
            done = False
            while not done:
                try:
                    n = int(n)
                    if n > 0:
                        self.n = n
                        done = True
                    else:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                except:
                    n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
        elif self.n > 0:
            print(f'   - n (Number of Cuts = n^2) = {self.n}')
        else:
            e = f"Invalid 'n' value Input of {self.n}\n"
            raise ValueError(e)
        
        
        if self.cuts is None:
            cut = input(f'   - Cut [1-{self.n**2},all] = ')
            done = False
            while not done:
                if cut == 'all':
                    self.cuts = range(1,self.n**2+1)
                    done = True
                elif cut in np.array(range(1,self.n**2+1)).astype(str):
                    self.cuts = [int(cut)]
                    done = True
                else:
                    cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
        elif self.cuts == 'all':
            print(f'   - Cut = all')
            self.cuts = range(1,self.n**2+1)
        elif self.cuts in range(1,self.n**2+1):
            print(f'   - Cut = {self.cuts}')
            self.cuts = [self.cuts]
        elif type(self.cuts) == list:
            print(f'   - Cut = {self.cuts}')
        else:
            e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
            raise ValueError(e)
        
        print('\n')

        if self.cut_time is None:
            if use_suggestions:
                self.cut_time = suggestions[0]
                print(f'   - Cut Batch Time = {self.cut_time}')
            else:
                cut_time = input(f"   - Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in cut_time:
                        self.cut_time = cut_time
                        done = True
                    else:
                        cut_time = input(f"      Invalid format! Cut Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Cut Batch Time = {self.cut_time}')
        
        if self.cut_mem is None:
            if use_suggestions:
                self.cut_mem = int(suggestions[1][:-1])
                print(f'   - Cut Mem/CPU = {self.cut_mem}G')
            else:
                cut_mem = input(f"   - Cut Mem/CPU ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try: 
                        cut_mem = int(cut_mem)
                        if 0<cut_mem < 500:
                            self.cut_mem = cut_mem
                            done=True
                        else:
                            cut_mem = input(f"      Invalid format! Cut Mem/CPU ({suggestions[1]} suggested) = ")
                    except:
                        if cut_mem[-1].lower() == 'g':
                            self.cut_mem = cut_mem[:-1]
                            done = True
                        else:
                            cut_mem = input(f"      Invalid format! Cut Mem/CPU ({suggestions[1]} suggested) = ")

        elif 0 < self.cut_mem < 500:
            print(f'   - Cut Mem/CPU = {self.cut_mem}G')
        else:
            e = f"Invalid Cut Mem/CPU Input of {self.cut_mem}\n"
            raise ValueError(e)
        
        
        if type(self.download_number) == int:
            cut_cpu = input(f"   - Cut Num CPUs [1-32] = ")
            done = False
            while not done:
                try:
                    cut_cpu = int(cut_cpu)
                    if 0 < cut_cpu < 33:
                        self.cut_cpu = cut_cpu
                        done = True
                    else:
                        cut_cpu = input(f"      Invalid format! Cut Num CPUs [1-32] = ")
                except:
                    cut_cpu = input(f"      Invalid format! Cut Num CPUs [1-32] = ")
        else:
            if suggestions[2] % int(self.cut_mem) != 0:
                self.cut_cpu = suggestions[2] // int(self.cut_mem) + 1
            else:
                self.cut_cpu = suggestions[2] // int(self.cut_mem)

            print(f'   - Cut Num CPUs Needed = {self.cut_cpu}')

        print('\n')

    def _predict_asteroids_properties(self,making_cuts,suggestions,use_suggestions):
        """
        Confirm asteroid prediction process properties. CPU is set directly
        (not derived from a mem/cpu ratio like cube/cut) since asteroid
        prediction parallelises via a fixed worker-process pool, matching
        reduce/search's convention rather than cube/cut's.

        predict_asteroids runs ahead of reduce() and can be the very first
        cut-consuming stage in a run (e.g. reduce=False, make_cuts=False --
        already-reduced data), so unlike reduce/calibrate/search it can't
        assume self.n/self.cuts were already resolved from 'all' by an
        earlier stage's properties method; resolves them itself here
        (mirroring _reduce_properties's own if not cutting: block) when
        make_cuts won't. Confirmed the hard way: leaving this out let
        self.cuts stay the literal string 'all', and `for cut in self.cuts`
        silently iterated its characters ('a', 'l', 'l') instead of raising.
        """

        if not making_cuts:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)


            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)

            print('\n')

        if self.predict_asteroids_time is None:
            if use_suggestions:
                self.predict_asteroids_time = suggestions[0]
                print(f'   - Predict Asteroids Batch Time = {self.predict_asteroids_time}')
            else:
                predict_asteroids_time = input(f"   - Predict Asteroids Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in predict_asteroids_time:
                        self.predict_asteroids_time = predict_asteroids_time
                        done = True
                    else:
                        predict_asteroids_time = input(f"      Invalid format! Predict Asteroids Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Predict Asteroids Batch Time = {self.predict_asteroids_time}')


        if self.predict_asteroids_cpu is None:
            if use_suggestions:
                self.predict_asteroids_cpu = int(suggestions[1])
                print(f'   - Predict Asteroids Num CPUs = {self.predict_asteroids_cpu}')
            else:
                predict_asteroids_cpu = input(f"   - Predict Asteroids Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        predict_asteroids_cpu = int(predict_asteroids_cpu)
                        if 0 < predict_asteroids_cpu < 33:
                            self.predict_asteroids_cpu = predict_asteroids_cpu
                            done = True
                        else:
                            predict_asteroids_cpu = input(f"      Invalid format! Predict Asteroids Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        predict_asteroids_cpu = input(f"      Invalid format! Predict Asteroids Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        elif 0 < self.predict_asteroids_cpu < 33:
            print(f'   - Predict Asteroids Num CPUs = {self.predict_asteroids_cpu}')
        else:
            e = f"Invalid Predict Asteroids CPUs Input of {self.predict_asteroids_cpu}\n"
            raise ValueError(e)

        if self.predict_asteroids_mem is None:
            if type(self.download_number) == int:
                predict_asteroids_mem = input(f"   - Predict Asteroids Mem/CPU = ")
                done = False
                while not done:
                    try:
                        predict_asteroids_mem = int(predict_asteroids_mem)
                        if 0<predict_asteroids_mem < 500:
                            self.predict_asteroids_mem = predict_asteroids_mem
                            done=True
                        else:
                            predict_asteroids_mem = input(f"      Invalid format! Predict Asteroids Mem/CPU = ")
                    except:
                        if predict_asteroids_mem[-1].lower() == 'g':
                            self.predict_asteroids_mem = predict_asteroids_mem[:-1]
                            done = True
                        else:
                            predict_asteroids_mem = input(f"      Invalid format! Predict Asteroids Mem/CPU = ")
            else:
                self.predict_asteroids_mem = np.ceil(suggestions[2]/self.predict_asteroids_cpu).astype(int)
                print(f'   - Predict Asteroids Mem/CPU Needed = {self.predict_asteroids_mem}')
        elif 0 < self.predict_asteroids_mem < 500:
            print(f'   - Predict Asteroids Mem/CPU = {self.predict_asteroids_mem}G')
        else:
            e = f"Invalid Predict Asteroids Mem/CPU Input of {self.predict_asteroids_mem}\n"
            raise ValueError(e)
        print('\n')

    def _reduce_properties(self,cutting,suggestions,use_suggestions):
        """
        Confirm reduction process properties.
        """

        if not cutting:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]  
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            print('\n')

        if self.reduce_time is None:
            if use_suggestions:
                self.reduce_time = suggestions[0]
                print(f'   - Reduce Batch Time = {self.reduce_time}')
            else:
                reduce_time = input(f"   - Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in reduce_time:
                        self.reduce_time = reduce_time
                        done = True
                    else:
                        reduce_time = input(f"      Invalid format! Reduce Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Reduce Batch Time = {self.reduce_time}')

        
        if self.reduce_cpu is None:
            if use_suggestions:
                self.reduce_cpu = int(suggestions[1])
                print(f'   - Reduce Num CPUs = {self.reduce_cpu}')
            else:
                reduce_cpu = input(f"   - Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        reduce_cpu = int(reduce_cpu)
                        if 0 < reduce_cpu < 33:
                            self.reduce_cpu = reduce_cpu
                            done = True
                        else:
                            reduce_cpu = input(f"      Invalid format! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        reduce_cpu = input(f"      Invalid format! Reduce Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        elif 0 < self.reduce_cpu < 33:
            print(f'   - Reduce Num CPUs = {self.reduce_cpu}')
        else:
            e = f"Invalid Reduce CPUs Input of {self.reduce_cpu}\n"
            raise ValueError(e)
        
        if self.reduce_mem is None:
            if type(self.download_number) == int:
                reduce_mem = input(f"   - Reduce Mem/CPU = ")
                done = False
                while not done:
                    try:
                        reduce_mem = int(reduce_mem)
                        if 0<reduce_mem < 500:
                            self.reduce_mem = reduce_mem
                            done=True
                        else:
                            reduce_mem = input(f"      Invalid format! Reduce Mem/CPU = ")
                    except:
                        if reduce_mem[-1].lower() == 'g':
                            self.reduce_mem = reduce_mem[:-1]
                            done = True
                        else:
                            reduce_mem = input(f"      Invalid format! Reduce Mem/CPU = ")
            else:
                self.reduce_mem = np.ceil(suggestions[2]/self.reduce_cpu).astype(int)
                print(f'   - Reduce Mem/CPU Needed = {self.reduce_mem}')
        elif 0 < self.reduce_mem < 500:
            print(f'   - Reduce Mem/CPU = {self.reduce_mem}G')
        else:
            e = f"Invalid Reduce Mem/CPU Input of {self.reduce_mem}\n"
            raise ValueError(e)
        print('\n')

    def _asteroid_lightcurves_properties(self,making_cuts_or_predicting,suggestions,use_suggestions):
        """
        Confirm asteroid lightcurve generation process properties. CPU is
        set directly (not derived from a mem/cpu ratio), matching
        reduce/predict_asteroids' convention.

        Like _predict_asteroids_properties, resolves self.n/self.cuts from
        'all' itself when neither make_cuts nor predict_asteroids will --
        asteroid_lightcurves can run standalone against already-reduced,
        already-predicted data (reduce=False, predict_asteroids=False),
        the same gap that let self.cuts stay the literal string 'all' and
        silently iterate its characters instead of raising.
        """

        if not making_cuts_or_predicting:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)


            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)

            print('\n')

        if self.asteroid_lightcurves_time is None:
            if use_suggestions:
                self.asteroid_lightcurves_time = suggestions[0]
                print(f'   - Asteroid Lightcurves Batch Time = {self.asteroid_lightcurves_time}')
            else:
                asteroid_lightcurves_time = input(f"   - Asteroid Lightcurves Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in asteroid_lightcurves_time:
                        self.asteroid_lightcurves_time = asteroid_lightcurves_time
                        done = True
                    else:
                        asteroid_lightcurves_time = input(f"      Invalid format! Asteroid Lightcurves Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Asteroid Lightcurves Batch Time = {self.asteroid_lightcurves_time}')


        if self.asteroid_lightcurves_cpu is None:
            if use_suggestions:
                self.asteroid_lightcurves_cpu = int(suggestions[1])
                print(f'   - Asteroid Lightcurves Num CPUs = {self.asteroid_lightcurves_cpu}')
            else:
                asteroid_lightcurves_cpu = input(f"   - Asteroid Lightcurves Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        asteroid_lightcurves_cpu = int(asteroid_lightcurves_cpu)
                        if 0 < asteroid_lightcurves_cpu < 33:
                            self.asteroid_lightcurves_cpu = asteroid_lightcurves_cpu
                            done = True
                        else:
                            asteroid_lightcurves_cpu = input(f"      Invalid format! Asteroid Lightcurves Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        asteroid_lightcurves_cpu = input(f"      Invalid format! Asteroid Lightcurves Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        elif 0 < self.asteroid_lightcurves_cpu < 33:
            print(f'   - Asteroid Lightcurves Num CPUs = {self.asteroid_lightcurves_cpu}')
        else:
            e = f"Invalid Asteroid Lightcurves CPUs Input of {self.asteroid_lightcurves_cpu}\n"
            raise ValueError(e)

        if self.asteroid_lightcurves_mem is None:
            if type(self.download_number) == int:
                asteroid_lightcurves_mem = input(f"   - Asteroid Lightcurves Mem/CPU = ")
                done = False
                while not done:
                    try:
                        asteroid_lightcurves_mem = int(asteroid_lightcurves_mem)
                        if 0<asteroid_lightcurves_mem < 500:
                            self.asteroid_lightcurves_mem = asteroid_lightcurves_mem
                            done=True
                        else:
                            asteroid_lightcurves_mem = input(f"      Invalid format! Asteroid Lightcurves Mem/CPU = ")
                    except:
                        if asteroid_lightcurves_mem[-1].lower() == 'g':
                            self.asteroid_lightcurves_mem = asteroid_lightcurves_mem[:-1]
                            done = True
                        else:
                            asteroid_lightcurves_mem = input(f"      Invalid format! Asteroid Lightcurves Mem/CPU = ")
            else:
                self.asteroid_lightcurves_mem = np.ceil(suggestions[2]/self.asteroid_lightcurves_cpu).astype(int)
                print(f'   - Asteroid Lightcurves Mem/CPU Needed = {self.asteroid_lightcurves_mem}')
        elif 0 < self.asteroid_lightcurves_mem < 500:
            print(f'   - Asteroid Lightcurves Mem/CPU = {self.asteroid_lightcurves_mem}G')
        else:
            e = f"Invalid Asteroid Lightcurves Mem/CPU Input of {self.asteroid_lightcurves_mem}\n"
            raise ValueError(e)
        print('\n')

    def _calibrate_properties(self, reducing, suggestions, use_suggestions):
        """
        Confirm flux calibration process properties.
        """
        if suggestions is None:
            suggestions = ['15:00', '8', 4]

        if not reducing:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)

            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1, self.n**2 + 1)
                        done = True
                    elif cut in np.array(range(1, self.n**2 + 1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1, self.n**2 + 1)
            elif self.cuts in range(1, self.n**2 + 1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)

            print('\n')

        if self.calibrate_time is None:
            if use_suggestions:
                self.calibrate_time = suggestions[0]
                print(f'   - Calibrate Batch Time = {self.calibrate_time}')
            else:
                calibrate_time = input(f"   - Calibrate Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in calibrate_time:
                        self.calibrate_time = calibrate_time
                        done = True
                    else:
                        calibrate_time = input(f"      Invalid format! Calibrate Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Calibrate Batch Time = {self.calibrate_time}')

        if self.calibrate_cpu is None:
            if use_suggestions:
                self.calibrate_cpu = int(suggestions[1])
                print(f'   - Calibrate Num CPUs = {self.calibrate_cpu}')
            else:
                calibrate_cpu = input(f"   - Calibrate Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        calibrate_cpu = int(calibrate_cpu)
                        if 0 < calibrate_cpu < 33:
                            self.calibrate_cpu = calibrate_cpu
                            done = True
                        else:
                            calibrate_cpu = input(f"      Invalid format! Calibrate Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        calibrate_cpu = input(f"      Invalid format! Calibrate Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        else:
            print(f'   - Calibrate Num CPUs = {self.calibrate_cpu}')

        if self.calibrate_mem is None:
            self.calibrate_mem = int(np.ceil(suggestions[2] / self.calibrate_cpu))
            print(f'   - Calibrate Mem/CPU = {self.calibrate_mem}G')
        else:
            print(f'   - Calibrate Mem/CPU = {self.calibrate_mem}G')

        print('\n')

    def _search_properties(self,cutting_reducing,suggestions,use_suggestions):
        """
        Confirm transient search process properties.
        """

        if not cutting_reducing:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            print('\n')

        if self.search_time is None:
            if use_suggestions:
                self.search_time = suggestions[0]
                print(f'   - Search Batch Time = {self.search_time}')
            else:
                search_time = input(f"   - Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in search_time:
                        self.search_time = search_time
                        done = True
                    else:
                        search_time = input(f"      Invalid format! Search Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Search Batch Time = {self.search_time}')


        if self.search_cpu is None:
            if use_suggestions:
                self.search_cpu = int(suggestions[1])
                print(f'   - Search Num CPUs = {self.search_cpu}')
            else:
                search_cpu = input(f"   - Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        search_cpu = int(search_cpu)
                        if 0 < search_cpu < 33:
                            self.search_cpu = search_cpu
                            done = True
                        else:
                            search_cpu = input(f"      Invalid format! Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        search_cpu = input(f"      Invalid format! Search Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        elif 0 < self.search_cpu < 33:
            print(f'   - Search Num CPUs = {self.search_cpu}')
        else:
            e = f"Invalid Search CPUs Input of {self.search_cpu}\n"
            raise ValueError(e) 
        
        if self.search_mem is None:
            if type(self.download_number) == int:
                search_mem = input(f"   - Search Mem/CPU = ")
                done = False
                while not done:
                    try:
                        search_mem = int(search_mem)
                        if 0<search_mem < 500:
                            self.search_mem = search_mem
                            done=True
                        else:
                            search_mem = input(f"      Invalid format! Search Mem/CPU = ")
                    except:
                        if search_mem[-1].lower() == 'g':
                            self.search_mem = search_mem[:-1]
                            done = True
                        else:
                            search_mem = input(f"      Invalid format! Search Mem/CPU = ")
            else:
                self.search_mem = np.ceil(suggestions[2]/self.search_cpu).astype(int)
                print(f'   - Search Mem/CPU Needed = {self.search_mem}')
        elif 0 < self.search_mem < 500:
            print(f'   - Search Mem/CPU = {self.search_mem}G')
        else:
            e = f"Invalid Search Mem/CPU Input of {self.search_mem}\n"
            raise ValueError(e)

        pattern = r'^\d+(\.\d+)?(sec|min|hr|day)s?$'
        if self.time_bins is None:
            if use_suggestions:
                self.time_bins = suggestions[3].split(',')
                print(f'   - Search Time Bin = {self.time_bins}')
            else:
                time_bins = input(f"   - Search Time Bins [#sec,#min,#hr,#day] ({suggestions[3]} suggested) = ")
                done = False
                while not done:
                    if ',' in time_bins:
                        time_bins = time_bins.split(',')
                    else:
                        time_bins = [time_bins]

                    if all(re.match(pattern, t) for t in time_bins):
                        self.time_bins = time_bins
                        done = True
                    else:
                        time_bins = input(f"      Invalid format! Search Time Bins [#sec,#min,#hr,#day] ({suggestions[3]} suggested) = ")
        
        else:
            if type(self.time_bins) == str:
                self.time_bins = self.time_bins.split(',')
                if all(re.match(pattern, t) for t in self.time_bins):
                    print(f'   - Search Time Bin = {self.time_bins}')
                else:
                    e = f'Invalid Search Time Bin Input of {self.time_bins}\n'
                    raise ValueError(e)
            
            elif all(re.match(pattern, t) for t in self.time_bins):
                print(f'   - Search Time Bins = {self.time_bins}')
            else:
                e = f'Invalid Search Time Bins Input of {self.time_bins}\n'
                raise ValueError(e)


        print('\n')
    
    def _plotting_properties(self,searching,suggestions,use_suggestions):
        """
        Confirm transient search process properties.
        """

        if not searching:
            if self.n is None:
                n = input('   - n (Number of Cuts = n^2) = ')
                done = False
                while not done:
                    try:
                        n = int(n)
                        if n > 0:
                            self.n = n
                            done = True
                        else:
                            n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
                    except:
                        n = input('      Invalid choice! n (Number of Cuts = n^2) =  ')
            elif self.n > 0:
                print(f'   - n (Number of Cuts = n^2) = {self.n}')
            else:
                e = f"Invalid 'n' value Input of {self.n}\n"
                raise ValueError(e)
            
            
            if self.cuts is None:
                cut = input(f'   - Cut [1-{self.n**2},all] = ')
                done = False
                while not done:
                    if cut == 'all':
                        self.cuts = range(1,self.n**2+1)
                        done = True
                    elif cut in np.array(range(1,self.n**2+1)).astype(str):
                        self.cuts = [int(cut)]
                        done = True
                    else:
                        cut = input(f'      Invalid choice! Cut [1-{self.n**2},all] =  ')
            elif self.cuts == 'all':
                print(f'   - Cut = all')
                self.cuts = range(1,self.n**2+1)
            elif self.cuts in range(1,self.n**2+1):
                print(f'   - Cut = {self.cuts}')
                self.cuts = [self.cuts]
            elif type(self.cuts) == list:
                print(f'   - Cut = {self.cuts}')
            else:
                e = f"Invalid Cut Input of {self.cuts} with 'n' of {self.n}\n"
                raise ValueError(e)
            
            
            print('\n')


        if self.plot_time is None:
            if use_suggestions:
                self.plot_time = suggestions[0]
                print(f'   - Plotting Batch Time = {self.plot_time}')
            else:
                plot_time = input(f"   - Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
                done = False
                while not done:
                    if ':' in plot_time:
                        self.plot_time = plot_time
                        done = True
                    else:
                        plot_time = input(f"      Invalid format! Plotting Batch Time ['h:mm:ss'] ({suggestions[0]} suggested) = ")
        else:
            print(f'   - Plotting Batch Time = {self.plot_time}')

        if self.plot_cpu is None:
            if use_suggestions:
                self.plot_cpu = int(suggestions[1])
                print(f'   - Plotting Num CPUs = {self.plot_cpu}')
            else:
                plot_cpu = input(f"   - Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                done = False
                while not done:
                    try:
                        plot_cpu = int(plot_cpu)
                        if 0 < plot_cpu < 33:
                            self.plot_cpu = plot_cpu
                            done = True
                        else:
                            plot_cpu = input(f"      Invalid format! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
                    except:
                        plot_cpu = input(f"      Invalid format! Plotting Num CPUs [1-32] ({suggestions[1]} suggested) = ")
        elif 0 < self.plot_cpu < 33:
            print(f'   - Plotting Num CPUs = {self.plot_cpu}')
        else:
            e = f"Invalid Plotting CPUs Input of {self.plot_cpu}\n"
            raise ValueError(e)
        
        
        if type(self.download_number) == int:
            plot_mem = input(f"   - Plot Mem/CPU = ")
            done = False
            while not done:
                try:
                    plot_mem = int(plot_mem)
                    if 0<plot_mem < 500:
                        self.plot_mem = plot_mem
                        done=True
                    else:
                        plot_mem = input(f"      Invalid format! Plot Mem/CPU = ")
                except:
                    if plot_mem[-1].lower() == 'g':
                        self.plot_mem = plot_mem[:-1]
                        done = True
                    else:
                        plot_mem = input(f"      Invalid format! Plot Mem/CPU = ")
        else:
            self.plot_mem = np.ceil(suggestions[2]/self.plot_cpu).astype(int)
            print(f'   - Plot Mem/CPU Needed = {self.plot_mem}')

        print('\n')
    
    def _overwrite_suggestions(self, make_cube, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot):

        options = []
        if make_cube:
            options.append('cube')
        if make_cuts:
            options.append('cut')
        if predict_asteroids:
            options.append('asteroids')
        if reduce:
            options.append('reduce')
        if asteroid_lightcurves:
            options.append('lightcurves')
        if calibrate:
            options.append('calibrate')
        if search:
            options.append('search')
        if plot:
            options.append('plot')

        done = False
        over = input(f'   - Overwrite any steps? [y,n,{str(options)[1:-1]}] = ')
        while not done:

            if over == 'y':
                self.overwrite = 'all'
                done = True
            elif over == 'n':
                self.overwrite = None
                done = True
            else:
                self.overwrite = (over.replace(' ','')).split(',')
                good = True
                for thing in self.overwrite:
                    if thing not in ['cube','cut','asteroids','reduce','lightcurves','calibrate','search','plot']:
                        good = False
                if good:
                    done = True
                else:
                    over = input(f"      Invalid choice! Overwrite any steps? [y,n,{str(options)[1:-1]}] = ")
        
        
        print('\n')
    
    def _write_config(self,download,make_cube, fix_wcs, make_cuts, predict_asteroids, reduce, asteroid_lightcurves, calibrate, search, plot,delete):

        config = {
            "base": {
                "sector": self.sector,
                "cam": self.cam,
                "ccd": self.ccd,
            }}

        if download:
            config['download'] = {
                'download' : True,
                'download_number' : self.download_number}

        if make_cube:
            config['make_cubes'] = {
                'make_cubes' : True,
                'cube job time' : self.cube_time,
                'cube job memory' : self.cube_mem,
                'cube job cpu' : self.cube_cpu}

        if fix_wcs:
            config['fix_wcs'] = {
                'fix_wcs' : True}

        if make_cuts | predict_asteroids | reduce | asteroid_lightcurves | calibrate | search | plot:
            config['cut_properties'] = {
                'n' : self.n,
                'cuts' : list(self.cuts)}

        if make_cuts:
            config['make_cuts'] = {
                'make_cuts' : True,
                'cut job time' : self.cut_time,
                'cut job memory' : self.cut_mem,
                'cut job cpu' : self.cut_cpu}

        if predict_asteroids:
            config['predict_asteroids'] = {
                'predict_asteroids' : True,
                'predict_asteroids job time' : self.predict_asteroids_time,
                'predict_asteroids job memory' : self.predict_asteroids_mem,
                'predict_asteroids job cpu' : self.predict_asteroids_cpu}

        if reduce:
            config['reduce'] = {
                'reduce' : True,
                'reduce job time' : self.reduce_time,
                'reduce job memory' : self.reduce_mem,
                'reduce job cpu' : self.reduce_cpu}

        if asteroid_lightcurves:
            config['asteroid_lightcurves'] = {
                'asteroid_lightcurves' : True,
                'asteroid_lightcurves job time' : self.asteroid_lightcurves_time,
                'asteroid_lightcurves job memory' : self.asteroid_lightcurves_mem,
                'asteroid_lightcurves job cpu' : self.asteroid_lightcurves_cpu}

        if calibrate:
            config['calibrate'] = {
                'calibrate' : True,
                'calibrate job time' : self.calibrate_time,
                'calibrate job memory' : self.calibrate_mem,
                'calibrate job cpu' : self.calibrate_cpu}

        if search:
            config['search'] = {
                'search' : True,
                'search job time' : self.search_time,
                'search job memory' : self.search_mem,
                'search job cpu' : self.search_cpu,
                'search detection mode' : self.detect_mode,
                'search time bins' : self.time_bins}

        if plot:
            config['plot'] = {
                'plot' : True,
                'plot job time' : self.plot_time,
                'plot job memory' : self.plot_mem,
                'plot job cpu' : self.plot_cpu}

        config['options'] = {
            'verbose' : self.verbose,
            'delete ffis' : delete}
        
        config_file = f'{self.working_path}/tessellate_config.txt'
        with open(config_file, "w") as f:
            for section, options in config.items():
                f.write(f"[{section}]\n")
                for key, value in options.items():
                    # Strings get quotes, bools/lists/ints stay as-is
                    # if isinstance(value, str):
                    #     f.write(f"{key} = '{value}'\n")
                    # else:
                    f.write(f"{key} = {value}\n")
                f.write("\n")

        
    def _reset_logs(self,make_cube,make_cuts,predict_asteroids,reduce,asteroid_lightcurves,calibrate,search,plot):
        """
        Reset slurm job logs in provided output path.
        """

        if input('Delete Past Job Logs? [y/n] :\n').lower() == 'y':
            if make_cube:
                os.system(f'rm -f {self.job_output_path}/tessellate_cubing_logs/*')
            if make_cuts:
                os.system(f'rm -f {self.job_output_path}/tessellate_cutting_logs/*')
            if predict_asteroids:
                os.system(f'rm -f {self.job_output_path}/tessellate_asteroid_prediction_logs/*')
            if reduce:
                os.system(f'rm -f {self.job_output_path}/tessellate_reduction_logs/*')
            if asteroid_lightcurves:
                os.system(f'rm -f {self.job_output_path}/tessellate_asteroid_lightcurves_logs/*')
            if calibrate:
                os.system(f'rm -f {self.job_output_path}/tessellate_calibration_logs/*')
            if search:
                os.system(f'rm -f {self.job_output_path}/tessellate_search_logs/*')
            if plot:
                os.system(f'rm -f {self.job_output_path}/tessellate_plotting_logs/*')


        print('\n')
        
    def download(self):
        """
        Download FFIs! 
        """

        from .dataprocessor import DataProcessor

        for cam in self.cam:
            for ccd in self.ccd:
                if len(glob(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/image_files/*ffic.fits')) > 1000:
                    print(f'Sector {self.sector} Cam {cam} CCD {ccd} data already downloaded!')
                    print('\n')
                else:
                    # tDownload = t()
                    data_processor = DataProcessor(sector=self.sector,data_path=self.data_path,verbose=self.verbose)
                    data_processor.download(cam=cam,ccd=ccd,number=self.download_number)
                    # os.system('clear')
                    # print(message)
                    # print(f'Sector {self.sector} Cam {cam} Ccd {ccd} Download Complete ({((t()-tDownload)/60):.2f} mins).')
                    print('\n')

#     def make_cube(self,overwrite=True):
#         """
#         Make Cube! 

#         Process : Generates a python script for generating cube, then generates and submits a slurm script to call the python script.
#         """

#         _Save_space(f'{self.working_path}/cubing_scripts')

#         # # -- Delete old scripts -- #
#         # os.system(f'rm -f {self.working_path}/cubing_scripts/S{self.sector}C*')

#         if overwrite & (self.overwrite is not None):
#             if (self.overwrite == 'all') | ('cube' in self.overwrite):
#                 delete_files('cubes',self.data_path,self.sector,self.n,self.cam,self.ccd,part=self.part)

#         for cam in self.cam:
#             for ccd in self.ccd: 
#                 print(_Print_buff(60,f'Making Cube for Sector{self.sector} Cam{cam} Ccd{ccd}'))
#                 print('\n')

#                 # -- Generate Cube Path -- #
#                 cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
#                 if os.path.exists(cube_check):
#                     print(f'Cam {cam} CCD {ccd} cube already exists!')
#                     print('\n')
#                 else:

#                     # -- Create python file for cubing-- # 
#                     print(f'Creating Cubing Python File for Sector{self.sector} Cam{cam}Ccd{ccd}')
#                     python_text = f"\
# from tessellate import DataProcessor\n\
# \n\
# processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
# processor.make_cube(cam={cam},ccd={ccd},part={self.part})\n\
# with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt', 'w') as file:\n\
#     file.write('Cubed!')"   
                
#                     with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py", "w") as python_file:
#                         python_file.write(python_text)

#                     # -- Create bash file to submit job -- #
#                     #print('Creating Cubing Batch File')
#                     batch_text = f'\
# #!/bin/bash\n\
# #\n\
# #SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cubing\n\
# #SBATCH --output={self.job_output_path}/tessellate_cubing_logs/%A_%x_job_output.txt\n\
# #SBATCH --error={self.job_output_path}/tessellate_cubing_logs/%A_%x_errors.txt\n\
# #\n\
# #SBATCH --ntasks=1\n\
# #SBATCH --time={self.cube_time}\n\
# #SBATCH --cpus-per-task={self.cube_cpu}\n\
# #SBATCH --mem-per-cpu={self.cube_mem}G\n\
# #SBATCH --account=oz335\n\
# \n\
# PYTHONUNBUFFERED=1\n\
# source {VENV_PATH}/bin/activate\n\
# python {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py'
#                     with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh", "w") as batch_file:
#                         batch_file.write(batch_text)

#                     # -- Submit job -- #
#                     #print('Submitting Cubing Batch File')
#                     os.system(f'sbatch {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh')
#                     print('\n')

#     def fix_wcs(self,cubing):
#         """
#         Calls the WCSfixer class to find the best reference image, PSF fit stars, and update WCS information.
#         """

#         from .WCSfixer import TessFixer

#         tf = TessFixer(sector=self.sector,data_path=self.data_path)
#         for cam in self.cam:
#             for ccd in self.ccd: 
#                 print(_Print_buff(60,f'Running WCSfixer for Sector{self.sector} Cam{cam} Ccd{ccd}'))
#                 print('\n')

#                 cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
#                 if not os.path.exists(cube_check):
#                     if not cubing:
#                         e = 'No Cube File Detected to Cut!\n'
#                         raise ValueError(e)
#                     else:
#                         tStart = t()
#                         l = self.cube_time.split(':')
#                         seconds = 1 * int(l[-1]) + 60 * int(l[-2])
#                         if len(l) == 3:
#                             seconds += 3600 * int(l[-3])

#                         if (len(self.cam)>1) | (len(self.ccd)>1): 
#                             seconds = 42300

#                         go = False
#                         message = 'Waiting for Cube'
#                         i = 0
#                         while not go:
#                             if t()-tStart > seconds + 3600:
#                                 print('Restarting Cubing')
#                                 print('\n')
#                                 self.make_cube(overwrite=False)
#                                 tStart = t()
#                             else:
#                                 if i > 0:
#                                     print(message, end='\r')
#                                     sleep(120)
#                                 if os.path.exists(cube_check):
#                                     go = True
#                                     print('\n')
#                                 else:
#                                     message += '.'
#                                     i += 1

#                 # -- Generate Cube Path -- #
#                 wcs_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/wcs/ref/polyfit_coeffs.txt'
#                 if os.path.exists(wcs_check):
#                     print(f'Cam {cam} CCD {ccd} wcs already fixed!')
#                 else:
#                     tf.run(cam,ccd)
#                 print('\n')

    def _cam_ccd_cube(self, cam, ccd, time=None):
            """
            Generates a python script for generating a single cube, then generates and
            submits a slurm script to call it. Returns the submitted job_id.
            """

            if time is None:
                time = self.cube_time

            # -- Create python file for cubing-- #
            print(f'Creating Cubing Python File for Sector{self.sector} Cam{cam}Ccd{ccd}')
            python_text = f"\
from tessellate import DataProcessor\n\
\n\
processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
processor.make_cube(cam={cam},ccd={ccd},part={self.part})\n\
with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt', 'w') as file:\n\
    file.write('Cubed!')"

            with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py", "w") as python_file:
                python_file.write(python_text)

            # -- Create bash file to submit job -- #
            batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cubing\n\
#SBATCH --output={self.job_output_path}/tessellate_cubing_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_cubing_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={time}\n\
#SBATCH --cpus-per-task={self.cube_cpu}\n\
#SBATCH --mem-per-cpu={self.cube_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.py'
            
            with open(f"{self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh", "w") as batch_file:
                batch_file.write(batch_text)

            # -- Submit job -- #
            submission = os.popen(f'sbatch {self.working_path}/cubing_scripts/S{self.sector}C{cam}C{ccd}_script.sh').read()
            job_id = submission.strip().split()[-1]
            print('\n')
            return job_id

    def make_cube(self,overwrite=True):
        """
        Make Cube!

        Process : Generates a python script for generating cube, then generates and submits a slurm script to call the python script.
        Returns a cube_status dict keyed by (cam,ccd), matching the structure produced by reduce().
        """

        _Save_space(f'{self.working_path}/cubing_scripts')

        cube_status = {}

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('cube' in self.overwrite):
                delete_files('cubes',self.data_path,self.sector,self.n,self.cam,self.ccd,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Making Cube for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')

                cube_status[(cam,ccd)] = {'status': None, 'job_id': None, 'job_time': None}

                # -- Generate Cube Path -- #
                cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
                if os.path.exists(cube_check):
                    print(f'Cam {cam} CCD {ccd} cube already exists!')
                    print('\n')
                    cube_status[(cam,ccd)]['status'] = 'COMPLETED'
                else:
                    job_id = self._cam_ccd_cube(cam,ccd)
                    cube_status[(cam,ccd)]['status'] = 'INCOMPLETE'
                    cube_status[(cam,ccd)]['job_id'] = job_id
                    cube_status[(cam,ccd)]['job_time'] = self.cube_time

        return cube_status

    def fix_wcs(self,cube_status):
        """
        Calls the WCSfixer class to find the best reference image, PSF fit stars, and update WCS information.

        cube_status : either
            - False : cubing is not currently running. If a cube already exists for a given
                      cam/ccd, proceed; otherwise raise an error.
            - dict  : as returned by make_cube(), keyed by (cam,ccd) with 'status'/'job_id'/
                      'job_time'. fix_wcs waits on incomplete cubing jobs (restarting on
                      timeout), and runs the WCS fix for each cam/ccd as soon as its cube
                      is ready, rather than waiting for all cubes to finish first.
        """

        from datetime import timedelta
        from .WCSfixer import TessFixer

        tf = TessFixer(sector=self.sector,data_path=self.data_path)

        def _run_wcs_fix(cam,ccd):
            print(_Print_buff(60,f'Running WCSfixer for Sector{self.sector} Cam{cam} Ccd{ccd}'))
            print('\n')

            wcs_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/wcs/ref/polyfit_coeffs.txt'
            if os.path.exists(wcs_check):
                print(f'Cam {cam} CCD {ccd} wcs already fixed!')
            else:
                tf.run(cam,ccd)
            print('\n')

        if cube_status is False:
            # ---- no cubing job running: cube must already exist ----
            for cam in self.cam:
                for ccd in self.ccd:
                    cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
                    if not os.path.exists(cube_check):
                        e = 'No Cube File Detected to Cut!\n'
                        raise ValueError(e)
                    _run_wcs_fix(cam,ccd)
            return

        # ---- cube_status is a dict: interleave waiting with WCS fixing ----
        waiting_status = {}
        for cam in self.cam:
            for ccd in self.ccd:
                cube_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/cubed.txt'
                if os.path.exists(cube_check):
                    waiting_status[(cam,ccd)] = 'COMPLETED'
                else:
                    waiting_status[(cam,ccd)] = 'INCOMPLETE'

        i = 0
        while len(waiting_status.keys()) > 0:

            for key in list(waiting_status.keys()):
                cam, ccd = key
                if waiting_status[key] == 'COMPLETED':
                    _run_wcs_fix(cam,ccd)
                    del(waiting_status[key])

                elif cube_status[key]['status'] == 'COMPLETED':
                    _run_wcs_fix(cam,ccd)
                    del(waiting_status[key])

                else:
                    job_id = cube_status[key]['job_id']
                    job_status = _Check_job_status(job_id)
                    # if job_status == 'FAILED':
                    #     print(f'Cubing Failed for Cam {cam} CCD {ccd}')
                    #     print('\n')
                    #     del(waiting_status[key])
                    if job_status == 'TIMEOUT':
                        parts = list(map(int, cube_status[key]['job_time'].split(':')))
                        if len(parts) == 3:
                            h, m, s = parts
                        else:
                            h = 0
                            m, s = parts

                        td = timedelta(hours=h, minutes=m, seconds=s)
                        td += timedelta(minutes=30)
                        total = int(td.total_seconds())
                        h = total // 3600
                        m = (total % 3600) // 60
                        s = total % 60
                        result = f"{h}:{m:02}:{s:02}"

                        print(f'Restarting Cubing for Cam {cam} CCD {ccd} with new time limit of {result}')
                        job_id = self._cam_ccd_cube(cam,ccd,time=result)
                        cube_status[key]['job_id'] = job_id
                        cube_status[key]['job_time'] = result

                    elif job_status == 'COMPLETED':
                        cube_status[key]['status'] = job_status
                        _run_wcs_fix(cam,ccd)
                        del(waiting_status[key])

                    elif job_status not in ['RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED']:
                        e = f'Job {job_id} for cubing of Cam {cam} CCD {ccd} has unexpected status: {job_status}\n'
                        raise ValueError(e)

            if len(waiting_status.keys()) > 0:
                print('Waiting for Cubes' + i*'.', end='\r')
                sleep(600)
                i += 1
                    

    def _get_catalogues(self,cam,ccd,base_path):
        """
        Access internet, find Gaia sources and save for reduction.
        """

        # print('Importing .dataprocessor, .catalog_queries')
        from .dataprocessor import DataProcessor
        from .localisation import CutWCS
        from .catalog_queries import create_external_var_cat, create_external_gaia_cat
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")
        # print('Done!')
        # print('\n')


        data_processor = DataProcessor(sector=self.sector,data_path=self.data_path,verbose=self.verbose)
        _,_,cutCentreCoords,cutRadPx = data_processor.find_cuts(cam=cam,ccd=ccd,n=self.n,plot=False)
        rad = cutRadPx + 2*60/21

        #image_path = glob(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/*ffic.fits')[0]

        l = self.cut_time.split(':')
        seconds = 1 * int(l[-1]) + 60 * int(l[-2])
        if len(l) == 3:
            seconds += 3600 * int(l[-3])

        if (len(self.cam)>1) | (len(self.ccd)>1): 
            seconds = 42300

        # -- Waits for cuts to be completed. If max time exceeded , return not done so cut process is restarted -- #
        gaia_ccd = None
        completed = []
        message = 'Waiting for Cuts'
        timeStart = t()
        done = True
        i = 0
        while len(completed) < len(self.cuts):
            if (t()-timeStart) > seconds + 600:
                done = False
                break
            else:
                if i > 0:
                    print(message, end='\r')
                    sleep(120)
                message += '.'
                for cut in self.cuts:
                    if cut not in completed:
                        save_path = f'{base_path}/Cut{cut}of{self.n**2}'
                        if os.path.exists(f'{save_path}/variable_catalog.csv') & os.path.exists(f'{save_path}/local_gaia_cat.csv'):
                            completed.append(cut)
                        elif os.path.exists(f'{save_path}/cut.txt'):

                            wcs = CutWCS(data_path=self.data_path,sector=self.sector,cam=cam,ccd=ccd,cut=cut,n=self.n)

                            #try:
                            print(f'Generating Catalogues {cut}')
                            if os.path.exists(f'{save_path}/local_gaia_cat.csv'):
                                print('--Gaia catalog already made, skipping.')
                            else:      # its time to move external_save_cat to tessellate, this import takes ages!!             
                                # cutPath = f'{save_path}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}.fits'

                                if gaia_ccd is None:
                                    gaia_ccd = pd.read_csv(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/ccd_gaia_cat.csv')

                                # doneGaia = False
                                # attempt = 1
                                # while not doneGaia:
                                    # try:
                                gaia_ccd['x'],gaia_ccd['y'] = wcs.all_world2pix(gaia_ccd.ra,gaia_ccd.dec,0)
                                gaia_cut = gaia_ccd[(gaia_ccd.x > -2)&
                                                    (gaia_ccd.x < cutRadPx*2+2)&
                                                    (gaia_ccd.y > -2)&
                                                    (gaia_ccd.y < cutRadPx*2+2)]
                                gaia_cut = gaia_cut.drop(columns=['x','y'])
                                gaia_cut.to_csv(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/local_gaia_cat.csv',index=False)
                                    

                                        # create_external_gaia_cat(centre=cutCentreCoords[cut-1],size=rad*2,wcs=wcs,save_path=save_path,maglim=19,verbose=self.verbose>1) # oversize radius by 2 arcmin in terms of tess pixels
                                        # create_external_gaia_cat(tpf=cutPath,save_path=save_path,maglim=19,verbose=self.verbose>1) # oversize radius by 2 arcmin in terms of tess pixels
                                    # doneGaia = True
                                    # except Exception as e:
                                    #     print(f"--GAIA Catalogue Attempt {attempt} failed with error: {e}")
                                    #     sleep(120)
                                    #     attempt += 1

                            
                            if os.path.exists(f'{save_path}/variable_catalog.csv'):
                                print('--Variable catalog already made, skipping.')
                            else:
                                rad2 = rad*21/60**2
                                doneVar = False
                                attempt = 1
                                while not doneVar:
                                    try:
                                        create_external_var_cat(center=cutCentreCoords[cut-1],
                                                                size=rad2,save_path=save_path,verbose=self.verbose>1) # This one queries in degrees!!!!
                                        doneVar = True
                                    except Exception as e:
                                        print(f"--Variable Catalogue Attempt {attempt} failed with error: {e}")
                                        sleep(120)
                                        attempt+=1

                            completed.append(cut)

                i += 1

        return done

    def make_cuts(self,overwrite=True):
        """
        Make cuts! 

            Cube_time_sug : dictates how long the cube process should run for, cubing is restarted if need be.
            Cut_time_sug : dictates how long the cut process should run for, cutting is restarted if need be.
        """

        _Save_space(f'{self.working_path}/cutting_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/cutting_scripts/S{self.sector}C*')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('cut' in self.overwrite):
                delete_files('cuts',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Making Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}')) 
                print('\n')
                
                for cut in self.cuts:
                    go = False
                    if self.part:
                        cut_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/cut.txt'
                        cut_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/cut.txt'
                        if (os.path.exists(cut_check1)) & (os.path.exists(cut_check2)):
                            print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
                            print('\n')
                        else:
                            go = True
                    else:
                        cut_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/cut.txt'
                        if os.path.exists(cut_check):
                            print(f'Cam {cam} CCD {ccd} cut {cut} already made!')
                            print('\n')
                        else:
                            go = True
                    if go:
                        # -- Create python file for cubing, cutting, reducing a cut-- # 
                        print(f'Creating Cutting Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
                        python_text = f"\
from tessellate import DataProcessor\n\
\n\
part = {self.part}\n\
processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
processor.make_cuts(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part)\n\
if not part:\n\
    with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/cut.txt', 'w') as file:\n\
        file.write('Cut!')"   

                        with open(f"{self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
                            python_file.write(python_text)

                        # -- Create bash file to submit job -- #
                        #print('Creating Cutting Batch File')
                        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Cutting\n\
#SBATCH --output={self.job_output_path}/tessellate_cutting_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_cutting_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.cut_time}\n\
#SBATCH --cpus-per-task={self.cut_cpu}\n\
#SBATCH --mem-per-cpu={self.cut_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

                        with open(f"{self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
                            batch_file.write(batch_text)

                        #print('Submitting Cutting Batch File')
                        os.system(f'sbatch {self.working_path}/cutting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')
                        print('\n')

                if self.part:
                    for i in range(2):
                        done = self._get_catalogues(cam=cam,ccd=ccd,base_path=f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part{i+1}')
                        if not done:
                            break
                else:
                    done = self._get_catalogues(cam=cam,ccd=ccd,base_path=f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}')
                
                print('\n')
                if not done:
                    print('Restarting Cutting')
                    print('\n')
                    self.make_cuts(overwrite=False)

    def predict_asteroids(self,overwrite=True):
        """
        Predict Asteroids!

        Predicts every catalogued minor planet (MPCORB) crossing each cut's
        footprint across its observing window -- runs ahead of reduce(),
        needing only the cut's sky footprint and frame times (both already
        available once make_cuts() has produced the cut's TPF).

        Returns a prediction_status dict (mirroring reduce()'s
        reduction_status) so asteroid_lightcurves() can wait for these
        jobs to actually finish before running, the same way
        calibrate() waits on reduction_status -- otherwise a lightcurves
        job can start (and silently skip, "run predict_asteroids first!")
        before its cut's prediction has actually written the ephemeris
        it depends on.
        """

        _Save_space(f'{self.working_path}/asteroid_prediction_scripts')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('asteroids' in self.overwrite):
                delete_files('asteroids',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        prediction_status = {}
        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Predicting Asteroids for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')

                for cut in self.cuts:
                    prediction_status[(cam, ccd, cut)] = {'status': None, 'job_id': None, 'job_time': None}
                    if self.part:
                        asteroids_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/asteroids.txt'
                        asteroids_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/asteroids.txt'
                        done = (os.path.exists(asteroids_check1)) & (os.path.exists(asteroids_check2))
                    else:
                        asteroids_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/asteroids.txt'
                        done = os.path.exists(asteroids_check)

                    if done:
                        print(f'Cam {cam} CCD {ccd} cut {cut} asteroids already predicted!')
                        print('\n')
                        prediction_status[(cam, ccd, cut)]['status'] = 'COMPLETED'
                    else:
                        job_id = self._cut_predict_asteroids(cam=cam,ccd=ccd,cut=cut)
                        prediction_status[(cam, ccd, cut)]['status'] = 'INCOMPLETE'
                        prediction_status[(cam, ccd, cut)]['job_id'] = job_id
                        prediction_status[(cam, ccd, cut)]['job_time'] = self.predict_asteroids_time

        return prediction_status

    def _cut_predict_asteroids(self,cam,ccd,cut,time=None):

        import subprocess

        # -- Create python file for predicting asteroids on a cut -- #
        print(f'Creating Asteroid Prediction Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import DataProcessor\n\
\n\
part = {self.part}\n\
processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
processor.predict_asteroids(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part)"

        with open(f"{self.working_path}/asteroid_prediction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Asteroid Prediction Batch File')
        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_PredictAsteroids\n\
#SBATCH --output={self.job_output_path}/tessellate_asteroid_prediction_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_asteroid_prediction_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.predict_asteroids_time if time is None else time}\n\
#SBATCH --cpus-per-task={self.predict_asteroids_cpu}\n\
#SBATCH --mem-per-cpu={self.predict_asteroids_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/asteroid_prediction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

        with open(f"{self.working_path}/asteroid_prediction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)

        result = subprocess.run(
            f'sbatch {self.working_path}/asteroid_prediction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh',
            shell=True, capture_output=True, text=True
        )

        job_id = result.stdout.strip().split()[-1]
        print(f'Submitted batch job {job_id}')
        print('\n')

        return job_id

    def _cut_reduce(self,cam,ccd,cut,time=None):

        import subprocess

         # -- Create python file for reducing a cut-- # 
        print(f'Creating Reduction Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import DataProcessor\n\
import os\n\
\n\
part={self.part}\n\
processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
processor.reduce(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part,injection={self.injection},injection_dir='{self._inj_dir}')\n\
if not part:\n\
    if os.path.exists('{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/{self._inj_dir}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Shifts.npy'):\n\
        with open(f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/{self._inj_dir}/reduced.txt', 'w') as file:\n\
            file.write('Reduced!')"   
            # file.write('Reduced with TESSreduce version {tr.__version__}.')"
        with open(f"{self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Reduction Batch File')
        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Reduction\n\
#SBATCH --output={self.job_output_path}/tessellate_reduction_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_reduction_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.reduce_time if time is None else time}\n\
#SBATCH --cpus-per-task={self.reduce_cpu}\n\
#SBATCH --mem-per-cpu={self.reduce_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

        with open(f"{self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)
                
        #print('Submitting Reduction Batch File')
        # os.system(f'sbatch {self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

        result = subprocess.run(
            f'sbatch {self.working_path}/reduction_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh',
            shell=True, capture_output=True, text=True
        )

        job_id = result.stdout.strip().split()[-1]
        print(f'Submitted batch job {job_id}')
        print('\n')

        return job_id

    def reduce(self,overwrite=True):
        """
        Reduce! 
        """

        _Save_space(f'{self.working_path}/reduction_scripts')

        reduction_status = {}

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/reduction_scripts/S{self.sector}C*')

        if (overwrite) & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('reduce' in self.overwrite):
                delete_files('reductions',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd: 
                print(_Print_buff(60,f'Reducing Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                for cut in self.cuts:
                    reduction_status[(cam, ccd, cut)] = {'status': None, 'job_id': None, 'job_time': None}
                    if self.part:
                        cut_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        cut_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        if not os.path.exists(cut_check1):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut} Part 1!\n'
                            raise ValueError(e)
                        elif not os.path.exists(cut_check2):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut} Part 2!\n'
                            raise ValueError(e)
                        
                        reduced_check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/{self._inj_dir}/reduced.txt'
                        reduced_check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/{self._inj_dir}/reduced.txt'
                        if (os.path.exists(reduced_check1))&(os.path.exists(reduced_check2)):
                            print(f'Cam {cam} CCD {ccd} Cut {cut} already reduced!')
                            print('\n')
                            reduction_status[(cam, ccd, cut)]['status'] = 'COMPLETED'
                        else:
                            job_id = self._cut_reduce(cam=cam,ccd=ccd,cut=cut)
                            reduction_status[(cam, ccd, cut)]['status'] = 'INCOMPLETE'
                            reduction_status[(cam, ccd, cut)]['job_id'] = job_id
                            reduction_status[(cam, ccd, cut)]['job_time'] = self.reduce_time
                        
                    else:
                        cut_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/local_gaia_cat.csv'
                        if not os.path.exists(cut_check):
                            e = f'No Source Catalogue Detected for Reduction of Cut {cut}!\n'
                            raise ValueError(e)
                    
                        reduced_check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/{self._inj_dir}/reduced.txt'
                        if os.path.exists(reduced_check):
                            print(f'Cam {cam} CCD {ccd} Cut {cut} already reduced!')
                            print('\n')
                            reduction_status[(cam, ccd, cut)]['status'] = 'COMPLETED'
                        else:
                            job_id = self._cut_reduce(cam=cam,ccd=ccd,cut=cut)
                            reduction_status[(cam, ccd, cut)]['status'] = 'INCOMPLETE'
                            reduction_status[(cam, ccd, cut)]['job_id'] = job_id
                            reduction_status[(cam, ccd, cut)]['job_time'] = self.reduce_time
                    
        return reduction_status

    def asteroid_lightcurves(self, prediction_status, overwrite=True):
        """
        Build Asteroid Lightcurves!

        Forced aperture + PSF photometry, pixel-phase detrending, star-
        contamination flagging, and shift-and-stack for every asteroid
        predict_asteroids() found crossing each cut, using the REDUCED
        flux cube -- runs after reduce(). Waits for predict_asteroids()'s
        own jobs to actually finish (via prediction_status, its returned
        status dict) before submitting a cut's lightcurves job, the same
        way calibrate() waits on reduction_status. Pass prediction_status
        =False to skip waiting entirely (predictions already exist / are
        being managed separately).
        """

        from datetime import timedelta

        _Save_space(f'{self.working_path}/asteroid_lightcurves_scripts')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('lightcurves' in self.overwrite):
                delete_files('asteroid_lightcurves',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        lightcurve_status = {}
        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Building Asteroid Lightcurves for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')

                for cut in self.cuts:
                    if self.part:
                        check1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/asteroid_lightcurves.txt'
                        check2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/asteroid_lightcurves.txt'
                        done = (os.path.exists(check1)) & (os.path.exists(check2))
                    else:
                        check = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/asteroid_lightcurves.txt'
                        done = os.path.exists(check)

                    if done:
                        print(f'Cam {cam} CCD {ccd} cut {cut} asteroid lightcurves already built!')
                        print('\n')
                    else:
                        lightcurve_status[(cam, ccd, cut)] = 'INCOMPLETE'

        i = 0
        while len(lightcurve_status.keys()) > 0:

            for key in list(lightcurve_status.keys()):
                cam, ccd, cut = key
                if prediction_status == False or prediction_status[key]['status'] == 'COMPLETED':
                    self._cut_asteroid_lightcurves(cam=cam,ccd=ccd,cut=cut)
                    del(lightcurve_status[key])

                else:
                    job_id = prediction_status[key]['job_id']
                    job_status = _Check_job_status(job_id)
                    if job_status == 'FAILED':
                        print(f'Asteroid Prediction Failed for Cam {cam} CCD {ccd} Cut {cut}')
                        print('\n')
                        del(lightcurve_status[key])
                    elif job_status == 'TIMEOUT':
                        parts = list(map(int, prediction_status[key]['job_time'].split(':')))
                        if len(parts) == 3:
                            h, m, s = parts
                        else:
                            h = 0
                            m, s = parts

                        td = timedelta(hours=h, minutes=m, seconds=s)
                        td += timedelta(minutes=30) # add 30 minutes to the job time
                        total = int(td.total_seconds())
                        h = total // 3600
                        m = (total % 3600) // 60
                        s = total % 60
                        result = f"{h}:{m:02}:{s:02}"

                        print(f'Restarting Asteroid Prediction for Cam {cam} CCD {ccd} Cut {cut} with new time limit of {result}')
                        job_id = self._cut_predict_asteroids(cam=cam,ccd=ccd,cut=cut,time=result)
                        prediction_status[key]['job_id'] = job_id
                        prediction_status[key]['job_time'] = result

                    elif job_status == 'COMPLETED':
                        prediction_status[key]['status'] = job_status

                    elif job_status not in ['RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED']:
                        e = f'Job {job_id} for asteroid prediction of Cam {cam} CCD {ccd} Cut {cut} has unexpected status: {job_status}\n'
                        raise ValueError(e)

            if prediction_status != False and len(lightcurve_status.keys()) > 0:
                print('Waiting for Asteroid Predictions' + i*'.', end='\r')
                sleep(120)
                i += 1

    # forced photometry cost scales directly with tracks x frames, and real per-cut track
    # counts can vary by an order of magnitude or more within a single sector (density
    # depends on ecliptic latitude -- confirmed on Sector 32: Cut57 at 54 tracks ran cleanly
    # within the secondary tier's suggested budget, while Cam1 Ccd3/Ccd4's cuts (~500-600+
    # tracks) needed roughly 4x that and OOM'd/TIMED OUT repeatedly at the plain suggestion)
    _LIGHTCURVES_BASELINE_TRACKS = 54

    def _lightcurves_resources_for_cut(self,cam,ccd,cut):
        """Scale asteroid_lightcurves' time/mem request by this cut's actual predicted track
        count, read from predict_asteroids' own already-completed output (available by the
        time this is called, since asteroid_lightcurves() only submits a cut's job once its
        prediction_status shows COMPLETED) -- rather than using one static, sector-wide guess
        for every cut regardless of how crowded it is. Falls back to the plain suggested
        time/cpu/mem if the ephemeris can't be read for any reason (e.g. genuinely zero
        tracks, or an unexpected layout)."""
        from .tools import load_table, table_exists

        base_time, cpu, base_mem = self.asteroid_lightcurves_time, self.asteroid_lightcurves_cpu, self.asteroid_lightcurves_mem
        try:
            cutFolder = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
            base = f'sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}'
            path = f'{cutFolder}/asteroids/{base}_Asteroids.csv'
            if not table_exists(path):
                return base_time, cpu, base_mem
            n_tracks = load_table(path)['designation'].nunique()
        except Exception:
            return base_time, cpu, base_mem

        scale = max(1.0, n_tracks / self._LIGHTCURVES_BASELINE_TRACKS)

        parts = list(map(int, base_time.split(':')))
        h, m, s = (0, parts[0], parts[1]) if len(parts) == 2 else parts
        total = int((h * 3600 + m * 60 + s) * scale)
        scaled_time = f"{total // 3600}:{(total % 3600) // 60:02}:{total % 60:02}"
        scaled_mem = int(np.ceil(base_mem * scale))

        return scaled_time, cpu, scaled_mem

    def _cut_asteroid_lightcurves(self,cam,ccd,cut):

        time, cpu, mem = self._lightcurves_resources_for_cut(cam,ccd,cut)

        # -- Create python file for building asteroid lightcurves on a cut -- #
        print(f'Creating Asteroid Lightcurves Python File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import DataProcessor\n\
\n\
part = {self.part}\n\
processor = DataProcessor(sector={self.sector},data_path='{self.data_path}',verbose=2)\n\
processor.asteroid_lightcurves(cam={cam},ccd={ccd},n={self.n},cut={cut},part=part)"

        with open(f"{self.working_path}/asteroid_lightcurves_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Asteroid Lightcurves Batch File')
        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_AsteroidLightcurves\n\
#SBATCH --output={self.job_output_path}/tessellate_asteroid_lightcurves_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_asteroid_lightcurves_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={time}\n\
#SBATCH --cpus-per-task={cpu}\n\
#SBATCH --mem-per-cpu={mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/asteroid_lightcurves_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

        with open(f"{self.working_path}/asteroid_lightcurves_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)

        #print('Submitting Asteroid Lightcurves Batch File')
        os.system(f'sbatch {self.working_path}/asteroid_lightcurves_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')
        print('\n')

    def _cut_calibrate(self, cam, ccd, cut):

        import subprocess

        print(f'Creating Calibration Script for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')

        python_text = f"\
import numpy as np\n\
from astropy.io import fits\n\
from astropy.wcs import WCS\n\
from tessellate import DataProcessor\n\
import os\n\
from tessellate.psf_flux_calibration import run_calibration, compute_detection_limits\n\
\n\
cut_folder = '{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'\n\
cal_folder = f'{{cut_folder}}/calibration'\n\
os.makedirs(cal_folder, exist_ok=True)\n\
wcs_path = '{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/wcs/ref/corrected.fits'\n\
ref_path = f'{{cut_folder}}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Ref.npy'\n\
flux_path = f'{{cut_folder}}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_ReducedFlux.npy'\n\
times_path = f'{{cut_folder}}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Times.npy'\n\
\n\
with fits.open(wcs_path) as f:\n\
    wcs = WCS(f[1].header)\n\
\n\
ref = np.load(ref_path)\n\
\n\
processor = DataProcessor(sector={self.sector}, data_path='{self.data_path}', verbose=0)\n\
cut_corners, _, _, _ = processor.find_cuts(cam={cam}, ccd={ccd}, n={self.n}, plot=False, verbose=0)\n\
cut_corner = cut_corners[{cut}-1]\n\
\n\
zp_ab, zp_err, _ = run_calibration(\n\
    ref, wcs,\n\
    sector={self.sector}, cam={cam}, ccd={ccd},\n\
    cut_corner=cut_corner,\n\
    n_jobs={self.calibrate_cpu},\n\
    plot=True,\n\
    savepath=cal_folder,\n\
)\n\
\n\
if os.path.exists(flux_path) and os.path.exists(times_path):\n\
    reduced_flux = np.load(flux_path)\n\
    time_array = np.load(times_path)\n\
    compute_detection_limits(\n\
        reduced_flux, time_array, zp_ab,\n\
        sector={self.sector}, cam={cam}, ccd={ccd},\n\
        cut_corner=cut_corner,\n\
        savepath=cal_folder,\n\
    )\n\
else:\n\
    print('ReducedFlux or Times not found — skipping detection limits.')\n\
\n\
with open(f'{{cal_folder}}/calibrated.txt', 'w') as file:\n\
    file.write(f'ZP_AB={{zp_ab:.6f}} E_ZP={{zp_err:.6f}}')"

        script_py = f'{self.working_path}/calibration_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'
        script_sh = script_py.replace('.py', '.sh')

        _Save_space(f'{self.working_path}/calibration_scripts')
        with open(script_py, 'w') as f:
            f.write(python_text)

        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Calibrate\n\
#SBATCH --output={self.job_output_path}/tessellate_calibration_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_calibration_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.calibrate_time}\n\
#SBATCH --cpus-per-task={self.calibrate_cpu}\n\
#SBATCH --mem-per-cpu={self.calibrate_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {script_py}'

        with open(script_sh, 'w') as f:
            f.write(batch_text)

        result = subprocess.run(
            f'sbatch {script_sh}',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f'sbatch failed for Cut {cut}:')
            print(f'  stdout: {result.stdout.strip()}')
            print(f'  stderr: {result.stderr.strip()}')
            print('\n')
            return None
        job_id = result.stdout.strip().split()[-1]
        print(f'Submitted batch job {job_id}')
        print('\n')
        return job_id

    # def calibrate(self, overwrite=True):
    #     """
    #     Flux Calibrate!  Derives a Gaia Rp AB zeropoint for each cut using
    #     PSF photometry on the reference image.
    #     """

    #     _Save_space(f'{self.working_path}/calibration_scripts')
    #     _Save_space(f'{self.job_output_path}/tessellate_calibration_logs')

    #     if overwrite and (self.overwrite is not None):
    #         if (self.overwrite == 'all') or ('calibrate' in self.overwrite):
    #             delete_files('calibrations', self.data_path, self.sector,
    #                          self.n, self.cam, self.ccd, self.cuts, part=self.part)

    #     for cam in self.cam:
    #         for ccd in self.ccd:
    #             print(_Print_buff(60, f'Calibrating Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
    #             print('\n')
    #             for cut in self.cuts:
    #                 cut_folder = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
    #                 ref_check = f'{cut_folder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Ref.npy'
    #                 if not os.path.exists(ref_check):
    #                     print(f'No reference image found for Cut {cut} - skipping (run reduce first).')
    #                     print('\n')
    #                     continue

    #                 cal_dir = f'{cut_folder}/calibration'
    #                 cal_done = (os.path.exists(f'{cal_dir}/calibrated.txt') and
    #                             os.path.exists(f'{cal_dir}/psf_calibration_zp.csv'))
    #                 if cal_done:
    #                     print(f'Cam {cam} CCD {ccd} Cut {cut} already calibrated!')
    #                     print('\n')
    #                 else:
    #                     self._cut_calibrate(cam=cam, ccd=ccd, cut=cut)

    def calibrate(self, reduction_status, overwrite=True):
        """
        Flux Calibrate!  Derives a Gaia Rp AB zeropoint for each cut using
        PSF photometry on the reference image.
        """

        from datetime import timedelta

        _Save_space(f'{self.working_path}/calibration_scripts')
        _Save_space(f'{self.job_output_path}/tessellate_calibration_logs')

        if overwrite and (self.overwrite is not None):
            if (self.overwrite == 'all') or ('calibrate' in self.overwrite):
                delete_files('calibrations', self.data_path, self.sector,
                            self.n, self.cam, self.ccd, self.cuts, part=self.part)

        calibrating_status = {}
        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60, f'Calibrating Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                for cut in self.cuts:
                    cut_folder = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                    cal_dir = f'{cut_folder}/calibration'
                    cal_done = (os.path.exists(f'{cal_dir}/calibrated.txt') and
                                os.path.exists(f'{cal_dir}/psf_calibration_zp.csv'))

                    if cal_done:
                        calibrating_status[(cam,ccd,cut)] = 'COMPLETED'
                    elif reduction_status == False:
                        ref_check = f'{cut_folder}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Ref.npy'
                        if not os.path.exists(ref_check):
                            e = f'No Reduced File Detected for Calibration of Cam {cam} Ccd {ccd} Cut {cut}!\n'
                            raise ValueError(e)
                        else:
                            calibrating_status[(cam,ccd,cut)] = 'INCOMPLETE'
                    else:
                        calibrating_status[(cam,ccd,cut)] = 'INCOMPLETE'

        i = 0
        while len(calibrating_status.keys()) > 0:

            for key in list(calibrating_status.keys()):
                cam,ccd,cut = key
                if calibrating_status[key] == 'COMPLETED':
                    print(f'Cam {cam} CCD {ccd} Cut {cut} already calibrated!')
                    print('\n')
                    del(calibrating_status[key])

                elif reduction_status == False or reduction_status[key]['status'] == 'COMPLETED':
                    self._cut_calibrate(cam=cam,ccd=ccd,cut=cut)
                    del(calibrating_status[key])

                else:
                    job_id = reduction_status[key]['job_id']
                    job_status = _Check_job_status(job_id)
                    if job_status == 'FAILED':
                        print(f'Reduction Failed for Cam {cam} CCD {ccd} Cut {cut}')
                        print('\n')
                        del(calibrating_status[key])
                    elif job_status == 'TIMEOUT':
                        parts = list(map(int, reduction_status[key]['job_time'].split(':')))
                        if len(parts) == 3:
                            h, m, s = parts
                        else:
                            h = 0
                            m, s = parts

                        td = timedelta(hours=h, minutes=m, seconds=s)
                        td += timedelta(minutes=30) # add 30 minutes to the job time
                        total = int(td.total_seconds())
                        h = total // 3600
                        m = (total % 3600) // 60
                        s = total % 60
                        result = f"{h}:{m:02}:{s:02}"

                        print(f'Restarting Reducing for Cam {cam} CCD {ccd} Cut {cut} with new time limit of {result}')
                        job_id = self._cut_reduce(cam=cam,ccd=ccd,cut=cut,time=result)
                        reduction_status[key]['job_id'] = job_id
                        reduction_status[key]['job_time'] = result

                    elif job_status == 'COMPLETED':
                        reduction_status[key]['status'] = job_status

                    elif job_status not in ['RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED']:
                        e = f'Job {job_id} for reduction of Cam {cam} CCD {ccd} Cut {cut} has unexpected status: {job_status}\n'
                        raise ValueError(e)

            if reduction_status != False and len(calibrating_status.keys()) > 0:
                print('Waiting for Reductions' + i*'.', end='\r')
                sleep(120)
                i += 1

    def _cut_transient_search(self,cam,ccd,cut):

        # -- Create python file for reducing a cut-- # 
        print(f'Creating Transient Search File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import Detector\n\
import os\n\
\n\
part = {self.part}\n\
\n\
if part:\n\
    path1 = '{self.data_path}/{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/{self._inj_dir}/detected_events.csv'\n\
    path2 = '{self.data_path}/{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/{self._inj_dir}/detected_events.csv'\n\
    if not os.path.exists(path1):\n\
        detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},injection={self.injection},part=1)\n\
        detector.transient_search(cut={cut},mode='{self.detect_mode}',time_bins={self.time_bins})\n\
    if not os.path.exists(path2):\n\
        detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},injection={self.injection},part=2)\n\
        detector.transient_search(cut={cut},mode='{self.detect_mode}',time_bins={self.time_bins})\n\
else:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},injection={self.injection},injection_dir='{self._inj_dir}')\n\
    detector.transient_search(cut={cut},mode='{self.detect_mode}',time_bins={self.time_bins})"   
                    
        with open(f"{self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Transient Search Batch File')
        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Search\n\
#SBATCH --output={self.job_output_path}/tessellate_search_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_search_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.search_time}\n\
#SBATCH --cpus-per-task={self.search_cpu}\n\
#SBATCH --mem-per-cpu={self.search_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

        with open(f"{self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)
                
        #print('Submitting Transient Search Batch File')
        os.system(f'sbatch {self.working_path}/detection_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

        print('\n')

    def transient_search(self, calibrating, overwrite=True, reduction_status=False):
        """
        Transient Search!
        """

        from datetime import timedelta

        _Save_space(f'{self.working_path}/detection_scripts')

        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('search' in self.overwrite):
                delete_files('search',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        def _already_searched(cam, ccd, cut):
            if self.part:
                save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/{self._inj_dir}'
                save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/{self._inj_dir}'
                return (os.path.exists(f'{save_path1}/detected_objects.csv') and
                        os.path.exists(f'{save_path2}/detected_objects.csv'))
            else:
                save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/{self._inj_dir}'
                return os.path.exists(f'{save_path}/detected_objects.csv')

        # ---- default behaviour: no reduction_status given, act immediately ----
        if reduction_status is False:
            for cam in self.cam:
                for ccd in self.ccd:
                    print(_Print_buff(60,f'Searching Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                    print('\n')
                    for cut in self.cuts:

                        if self.part:
                            save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}/{self._inj_dir}'
                            save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}/{self._inj_dir}'
                            if (os.path.exists(f'{save_path1}/detected_objects.csv')) & (os.path.exists(f'{save_path2}/detected_objects.csv')):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                print('\n')
                            elif (os.path.exists(f'{save_path1}/reduced.txt')) & (os.path.exists(f'{save_path2}/reduced.txt')):
                                self._cut_transient_search(cam,ccd,cut)
                            elif not os.path.exists(f'{save_path1}/reduced.txt'):
                                if calibrating:
                                    print(f'No Reduction Files Detected for Search of Cut {cut} Part 1! Skipping!')
                                    print('\n')
                                else:
                                    e = f'No Reduced File Detected for Search of Cut {cut} Part 1!\n'
                                    raise ValueError(e)
                            else:
                                if calibrating:
                                    print(f'No Reduction Files Detected for Search of Cut {cut} Part 2! Skipping!')
                                    print('\n')
                                else:
                                    e = f'No Reduced File Detected for Search of Cut {cut} Part 2!\n'
                                    raise ValueError(e)
                        else:
                            save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}/{self._inj_dir}'
                            if os.path.exists(f'{save_path}/detected_objects.csv'):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                                print('\n')
                            elif not os.path.exists(f'{save_path}/sector{self.sector}_cam{cam}_ccd{ccd}_cut{cut}_of{self.n**2}_Ref.npy'):
                                if calibrating:
                                    print(f'No Reduction Files Detected for Search of Cam {cam} Ccd {ccd} Cut {cut}! Skipping!')
                                    print('\n')
                                else:
                                    e = f'No Reduction Files Detected for Search of Cam {cam} Ccd {ccd} Cut {cut}!\n'
                                    raise ValueError(e)
                            else:
                                self._cut_transient_search(cam,ccd,cut)
            return

        # ---- reduction_status given: wait on reduction jobs like calibrate() does ----
        searching_status = {}
        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Searching Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                for cut in self.cuts:
                    if _already_searched(cam, ccd, cut):
                        searching_status[(cam,ccd,cut)] = 'COMPLETED'
                    else:
                        searching_status[(cam,ccd,cut)] = 'INCOMPLETE'

        i = 0
        while len(searching_status.keys()) > 0:

            for key in list(searching_status.keys()):
                cam, ccd, cut = key
                if searching_status[key] == 'COMPLETED':
                    print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
                    print('\n')
                    del(searching_status[key])

                elif reduction_status[key]['status'] == 'COMPLETED':
                    self._cut_transient_search(cam,ccd,cut)
                    del(searching_status[key])

                else:
                    job_id = reduction_status[key]['job_id']
                    job_status = _Check_job_status(job_id)
                    if job_status == 'FAILED':
                        print(f'Reduction Failed for Cam {cam} CCD {ccd} Cut {cut}')
                        print('\n')
                        del(searching_status[key])
                    elif job_status == 'TIMEOUT':
                        parts = list(map(int, reduction_status[key]['job_time'].split(':')))
                        if len(parts) == 3:
                            h, m, s = parts
                        else:
                            h = 0
                            m, s = parts

                        td = timedelta(hours=h, minutes=m, seconds=s)
                        td += timedelta(minutes=30)
                        total = int(td.total_seconds())
                        h = total // 3600
                        m = (total % 3600) // 60
                        s = total % 60
                        result = f"{h}:{m:02}:{s:02}"

                        print(f'Restarting Reducing for Cam {cam} CCD {ccd} Cut {cut} with new time limit of {result}')
                        job_id = self._cut_reduce(cam=cam,ccd=ccd,cut=cut,time=result)
                        reduction_status[key]['job_id'] = job_id
                        reduction_status[key]['job_time'] = result

                    elif job_status == 'COMPLETED':
                        reduction_status[key]['status'] = job_status

                    elif job_status not in ['RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED']:
                        e = f'Job {job_id} for reduction of Cam {cam} CCD {ccd} Cut {cut} has unexpected status: {job_status}\n'
                        raise ValueError(e)

            if len(searching_status.keys()) > 0:
                print('Waiting for Reductions' + i*'.', end='\r')
                sleep(120)
                i += 1

        


    # def transient_search(self,reduction_status,overwrite=True):
    #     """
    #     Transient Search!
    #     """

    #     from datetime import timedelta

    #     _Save_space(f'{self.working_path}/detection_scripts')

    #     # # -- Delete old scripts -- #
    #     # os.system(f'rm -f {self.working_path}/detection_scripts/S{self.sector}C*')

    #     if overwrite & (self.overwrite is not None):
    #         if (self.overwrite == 'all') | ('search' in self.overwrite):
    #             delete_files('search',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

    #     cutting_status = {}
    #     for cam in self.cam:
    #         for ccd in self.ccd:
    #             print(_Print_buff(60,f'Searching Cut(s) for Sector{self.sector} Cam{cam} Ccd{ccd}'))
    #             print('\n')
    #             for cut in self.cuts:
    #                 if self.part:
    #                     save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
    #                     save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
    #                     if (os.path.exists(f'{save_path1}/detected_objects.csv')) & (os.path.exists(f'{save_path2}/detected_objects.csv')):
    #                         cutting_status[(cam,ccd,cut)] = 'COMPLETED'
    #                     elif (os.path.exists(f'{save_path1}/reduced.txt')) & (os.path.exists(f'{save_path2}/reduced.txt')):
    #                         cutting_status[(cam,ccd,cut)] = 'INCOMPLETE'
    #                     elif not os.path.exists(f'{save_path1}/reduced.txt'):
    #                         e = f'No Reduced File Detected for Search of Cut {cut} Part 1!\n'
    #                         raise ValueError(e)
    #                     else:
    #                         e = f'No Reduced File Detected for Search of Cut {cut} Part 2!\n'
    #                         raise ValueError(e)
    #                 else:
    #                     save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
    #                     if os.path.exists(f'{save_path}/detected_objects.csv'):
    #                         cutting_status[(cam,ccd,cut)] = 'COMPLETED'               
    #                     elif os.path.exists(f'{save_path}/reduced.txt'):
    #                         cutting_status[(cam,ccd,cut)] = 'INCOMPLETE'
    #                     elif reduction_status == False:
    #                         e = f'No Reduced File Detected for Search of Cam {cam} Ccd {ccd} Cut {cut}!\n'
    #                         raise ValueError(e)
    #                     else:
    #                         cutting_status[(cam,ccd,cut)] = 'INCOMPLETE'

    #     i = 0 
    #     while len(cutting_status.keys()) > 0:

    #         for key in list(cutting_status.keys()):
    #             cam,ccd,cut = key
    #             if cutting_status[key] == 'COMPLETED':
    #                 print(f'Cam {cam} CCD {ccd} Cut {cut} already searched!')
    #                 print('\n')
    #                 del(cutting_status[key])

    #             elif reduction_status == False or reduction_status[key]['status'] == 'COMPLETED':
    #                 self._cut_transient_search(cam,ccd,cut)
    #                 del(cutting_status[key])
    #                 # if reduction_status != False:
    #                 #     del(reduction_status[key])

    #             else:
    #                 job_id = reduction_status[key]['job_id']
    #                 job_status = _Check_job_status(job_id)
    #                 if job_status == 'FAILED':
    #                     print(f'Reduction Failed for Cam {cam} CCD {ccd} Cut {cut}')
    #                     print('\n')
    #                     del(cutting_status[key])
    #                 elif job_status == 'TIMEOUT':
    #                     parts = list(map(int, reduction_status[key]['job_time'].split(':')))
    #                     if len(parts) == 3:
    #                         h, m, s = parts
    #                     else:
    #                         h = 0
    #                         m, s = parts
                        
    #                     td = timedelta(hours=h, minutes=m, seconds=s)
    #                     td += timedelta(minutes=30) # add 30 minutes to the job time
    #                     total = int(td.total_seconds())
    #                     h = total // 3600
    #                     m = (total % 3600) // 60
    #                     s = total % 60
    #                     result = f"{h}:{m:02}:{s:02}"

    #                     print(f'Restarting Reducing for Cam {cam} CCD {ccd} Cut {cut} with new time limit of {result}')
    #                     job_id = self._cut_reduce(cam=cam,ccd=ccd,cut=cut,time=result)
    #                     reduction_status[key]['job_id'] = job_id
    #                     reduction_status[key]['job_time'] = result

    #                 elif job_status == 'COMPLETED':
    #                     reduction_status[key]['status'] = job_status

    #                 elif job_status not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:
    #                     e = f'Job {job_id} for reduction of Cam {cam} CCD {ccd} Cut {cut} has unexpected status: {job_status}\n'
    #                     raise ValueError(e)

            # if reduction_status != False and len(cutting_status.keys()) > 0: 
            #     print('Waiting for Reductions' + i*'.', end='\r')
            #     sleep(600)
            #     i += 1
    
        
        
            



    def _cut_transient_plot(self,cam,ccd,cut):

        # -- Create python file for reducing a cut-- # 
        print(f'Creating Transient Plotting File for Sector{self.sector} Cam{cam} Ccd{ccd} Cut{cut}')
        python_text = f"\
from tessellate import Detector\n\
\n\
part = {self.part}\n\
if part:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=1)\n\
    detector.plot_ALL(cut={cut},lower=3)\n\
    detector.lc_ALL(cut={cut},lower=3)\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n},part=2)\n\
    detector.lc_ALL(cut={cut},lower=3)\n\
else:\n\
    detector = Detector(sector={self.sector},data_path='{self.data_path}',cam={cam},ccd={ccd},n={self.n})\n\
    detector.plot_ALL(cut={cut},lower=3)\n\
    detector.lc_ALL(cut={cut},lower=3)"   
                    
        with open(f"{self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py", "w") as python_file:
            python_file.write(python_text)

        # -- Create bash file to submit job -- #
        #print('Creating Transient Search Batch File')
        batch_text = f'\
#!/bin/bash\n\
#\n\
#SBATCH --job-name=TESS_S{self.sector}_Cam{cam}_Ccd{ccd}_Cut{cut}_Plotting\n\
#SBATCH --output={self.job_output_path}/tessellate_plotting_logs/%A_%x_job_output.txt\n\
#SBATCH --error={self.job_output_path}/tessellate_plotting_logs/%A_%x_errors.txt\n\
#\n\
#SBATCH --ntasks=1\n\
#SBATCH --time={self.plot_time}\n\
#SBATCH --cpus-per-task={self.plot_cpu}\n\
#SBATCH --mem-per-cpu={self.plot_mem}G\n\
#SBATCH --account=oz335\n\
\n\
PYTHONUNBUFFERED=1\n\
source {VENV_PATH}/bin/activate\n\
python {self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.py'

        with open(f"{self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh", "w") as batch_file:
            batch_file.write(batch_text)
                
        #print('Submitting Transient Search Batch File')
        os.system(f'sbatch {self.working_path}/plotting_scripts/S{self.sector}C{cam}C{ccd}C{cut}_script.sh')

        print('\n')

    def transient_plot(self,searching=False,overwrite=True): # here
        """
        Transient Search!
        """

        _Save_space(f'{self.working_path}/plotting_scripts')

        # # -- Delete old scripts -- #
        # os.system(f'rm -f {self.working_path}/plotting_scripts/S{self.sector}C*')
        
        if overwrite & (self.overwrite is not None):
            if (self.overwrite == 'all') | ('plot' in self.overwrite):
                delete_files('plot',self.data_path,self.sector,self.n,self.cam,self.ccd,self.cuts,part=self.part)

        for cam in self.cam:
            for ccd in self.ccd:
                print(_Print_buff(60,f'Transient Plotting for Sector{self.sector} Cam{cam} Ccd{ccd}'))
                print('\n')
                if not searching:
                    for cut in self.cuts:
                        if self.part:
                            save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                            save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                            if (os.path.exists(f'{save_path1}/lcs.zip')) & (os.path.exists(f'{save_path2}/lcs.zip')):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                print('\n')
                            elif (os.path.exists(f'{save_path1}/detected_objects.csv'))&(os.path.exists(f'{save_path2}/detected_objects.csv')):
                                self._cut_transient_plot(cam,ccd,cut)
                            elif not os.path.exists(f'{save_path1}/detected_objects.csv'):
                                e = f'No Event File Detected for Plotting of Cut {cut} Part 1!\n'
                                raise ValueError(e)
                            else:
                                e = f'No Event File Detected for Plotting of Cut {cut} Part 2!\n'
                                raise ValueError(e)
                        else:
                            save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                            if os.path.exists(f'{save_path}/lcs.zip'):
                                print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                print('\n')
                            elif os.path.exists(f'{save_path}/detected_objects.csv'):
                                self._cut_transient_plot(cam,ccd,cut)
                            else:
                                e = f'No Event File Detected for Plotting of Cut {cut}!\n'
                                print(e)
                                #raise ValueError(e)
                            
                        
                else:
                    completed = []
                    message = 'Waiting for Search'

                    tStart = t()
                    l = self.search_time.split(':')
                    seconds = 1 * int(l[-1]) + 60 * int(l[-2])
                    if len(l) == 3:
                        seconds += 3600 * int(l[-3])
                    else:
                        l.insert(0,0)

                    if (len(self.cam)>1) | (len(self.ccd)>1): 
                        seconds = 42300
                        
                    i = 0
                    while len(completed) < len(self.cuts):
                        if t()-tStart > seconds + 600:
                            print('Restarting Search')
                            print('\n')
                            self.search_time = f'{int(l[0])+1}:{l[1]}:{l[2]}'
                            self.transient_search(False,overwrite=False)
                            tStart = t()
                        else:
                            if i > 0:
                                print(message, end='\r')
                                sleep(120)
                            for cut in self.cuts:
                                if cut not in completed:
                                    if self.part:
                                        save_path1 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part1/Cut{cut}of{self.n**2}'
                                        save_path2 = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Part2/Cut{cut}of{self.n**2}'
                                        if (os.path.exists(f'{save_path1}/lcs.zip')) & (os.path.exists(f'{save_path2}/lcs.zip')):
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} plots already made!')
                                            print('\n')
                                        elif (os.path.exists(f'{save_path1}/detected_events.csv'))&(os.path.exists(f'{save_path2}/detected_events.csv')):
                                            self._cut_transient_plot(cam,ccd,cut)
                                            completed.append(cut)
                                    else:
                                        save_path = f'{self.data_path}/Sector{self.sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{self.n**2}'
                                        if os.path.exists(f'{save_path}/lcs.zip'):
                                            completed.append(cut)
                                            print(f'Cam {cam} CCD {ccd} Cut {cut} already plotted!')
                                            print('\n')
                                        elif os.path.exists(f'{save_path}/detected_events.csv'):
                                            self._cut_transient_plot(cam,ccd,cut)
                                            completed.append(cut)
                            i+=1

