import numpy as np
import os
import pandas as pd

class CutWCS():

    def __init__(self,data_path,sector,cam,ccd,cut,n,overlap=10):

        from astropy.io import fits
        from astropy.wcs import WCS
        from glob import glob

        self.data_path = data_path
        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.cut = cut
        self.n = n
        self.overlap = overlap

        # wcs_folder = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Cut{cut}of{self.n**2}/wcs_info'
        # folder = glob(f'{wcs_folder}/*corrected.fits')
        wcs_path = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/wcs/ref/corrected.fits'

        self.tesscut_offset = np.array([0,0])
        self._get_corner()

        if os.path.exists(wcs_path):
            self.wcs_path = wcs_path
            hdu = fits.open(self.wcs_path)
            self.wcs = WCS(hdu[1].header)
            self._get_median_offsets()
        else:
            wcs_folder = f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/Cut{cut}of{self.n**2}/wcs_info'
            if len(glob(f'{wcs_folder}/*corrected.fits')) > 0:
                self.wcs_path = glob(f'{wcs_folder}/*corrected.fits')[0]
                if os.path.exists(f'{wcs_folder}/tesscut_offset.npy'):
                    self.tesscut_offset = np.load(f'{wcs_folder}/tesscut_offset.npy')
            else:    
                self.wcs_path = glob(f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/*wcs.fits')[0]  # temporary

            hdu = fits.open(self.wcs_path)
            self.wcs = WCS(hdu[1].header)

            self.dx = self.dy = 0

        self.corner += self.tesscut_offset

    def _get_corner(self):

        intervals = 2048 / self.n

        cut_cornersX = [44 + i * intervals for i in range(self.n)]
        cut_cornersY = [i * intervals for i in range(self.n)]

        cut_corners = np.meshgrid(cut_cornersX, cut_cornersY)
        cut_corners = np.floor(
            np.stack((cut_corners[0], cut_corners[1]), axis=2).reshape(self.n**2, 2)
        )

        idx = np.arange(self.n * self.n)
        cols = idx % self.n
        rows = idx // self.n

        cut_corners[cols == self.n - 1, 0] -= self.overlap
        cut_corners[(cols != 0) & (cols != self.n - 1), 0] -= self.overlap // 2

        cut_corners[rows == self.n - 1, 1] -= self.overlap
        cut_corners[(rows != 0) & (rows != self.n - 1), 1] -= self.overlap // 2

        self.corner = cut_corners[self.cut - 1]

    def _get_median_offsets(self):

        sources = pd.read_csv(f'{self.data_path}/Sector{self.sector}/Cam{self.cam}/Ccd{self.ccd}/wcs/ref/ccd_sourcefits.csv')
        cut_size = 2048/self.n
        cut_sources = sources[(sources.xPSF>self.corner[0])&(sources.xPSF<self.corner[0]+cut_size)&
                              (sources.yPSF>self.corner[1])&(sources.yPSF<self.corner[1]+cut_size)]
        
        xFinal,yFinal = self.wcs.all_world2pix(cut_sources.ra,cut_sources.dec,0)

        self.dx = np.nanmedian(cut_sources.xPSF-xFinal)
        self.dy = np.nanmedian(cut_sources.yPSF-yFinal)

        del(sources)
        del(cut_sources)

    def all_world2pix(self,ra,dec,origin=0):

        """
        Convert RA and Dec to pixel coordinates for the cut WCS.
        """
        x, y = self.wcs.all_world2pix(ra, dec, origin)
        return x - self.corner[0] + self.dx, y - self.corner[1] + self.dy
    
    def all_pix2world(self, x, y, origin=0):
        """
        Convert pixel coordinates to RA and Dec for the cut WCS.
        """
        ra, dec = self.wcs.all_pix2world(x + self.corner[0] - self.dx, y + self.corner[1] - self.dy, origin)
        return ra, dec

    def __str__(self):
        return (
            str(self.wcs)
            + '\n\nNote: this is the WCS information for the full CCD. '
            + 'The cut WCS results add the cut corner offset to the pixel coordinates.'
        )

    def __repr__(self):
        return self.__str__()
    

    
class PSF_Fitter():
    def __init__(self,shape,PRF,verbose=False,central_shape=3,function='sq'):        
        """
        x   :    x dimension of psf kernel
        y   :    y dimension of psf kernel
        function : 'sq'= (image-psf)**2
                    'abs'= abs(image-psf)
                    'weight' = (image-psf)**2 * sqrt(image)
        """
        
        self.shape = shape
        self.source_x = 0         # offset of source from centre
        self.source_y = 0         # offset of source from centre
        self.verbose = verbose
        self.function = function

        # -- Finds centre of kernel -- #
        self.cent=int(self.shape/2.-0.5)
        self.mask = slice(self.cent-central_shape//2, self.cent+central_shape//2+1), slice(self.cent-central_shape//2, self.cent+central_shape//2+1)

        self.prf = PRF # initialise psf kernel (unnecessary)    

    def source(self,shiftx=0,shifty=0):

        centx_s = self.cent + shiftx    # source centre
        centy_s = self.cent + shifty

        psf = self.prf.locate(centx_s,centy_s, (self.shape,self.shape))
        self.psf = psf/np.nansum(psf)
        

    def minimizer(self, coeff, image):
        self.source_x = coeff[0]
        self.source_y = coeff[1]
        
        # -- generate psf -- #
        self.source(shiftx=self.source_x, shifty=self.source_y)
        
        # -- calculate residuals -- #
        diff = image - self.psf  # no abs() needed now
        # diff[image==0] /= 5      # keep your edge downweighting if you want

        if self.function == 'sq':
            residual = np.nansum(diff[self.mask]**2)#*np.sqrt(image[self.mask]))  # absolute!
        elif self.function == 'abs':
            residual = np.nansum(np.abs(diff[self.mask]))
        elif self.function == 'weight':
            residual = np.nansum(diff[self.mask]**2 * np.sqrt(image[self.mask]))

        return residual * 1e6
        
    def fit_psf(self,image,limx=3,limy=3):
        """
        Fit the PSF. Limx,y dictates bounds for position of the source
        """
        from scipy.optimize import minimize
        from copy import deepcopy

        image = deepcopy(image)

        image -= np.nanmedian(image)    # ensure background is calibrated ish
        image /= np.nansum(image[self.mask])    # normalise the image
        
        coeff = [self.source_x,self.source_y]
        lims = [[-limx,limx],[-limy,limy]]
        
        # -- Optimize -- #
        res = minimize(self.minimizer, coeff, args=image, method='Powell',bounds=lims)

        self.psf_fit = res




# def model(snr, a, b, c):
#     return a * snr**(-b) + c

def model_quadrature(snr, a, b, c):
    """
    Quadrature combination of a power-law statistical term and a fixed
    floor term.
    """
    return np.sqrt((a * snr**(-b))**2 + c**2)

def log_model_quadrature(snr, log_a, b, log_c):
    a = 10**log_a
    c = 10**log_c
    return np.log10(model_quadrature(snr, a, b, c))

def predict_median(f,snr):
    return float(f(snr))
    
def gen_and_fit_source(snr, shift, image_size, prf):

    noise_sigma = 1
    npix = 9
    exptime_s = 600  # Exposure time used to go from counts/second to raw counts in order to incorporate Poisson shot noise error 

    centx_s = image_size // 2 + shift[0]
    centy_s = image_size // 2 + shift[1]

    psf = prf.locate(centx_s, centy_s, (image_size, image_size))
    psf /= np.nansum(psf[image_size//2-1:image_size//2+2, image_size//2-1:image_size//2+2])

    # invert snr = flux / sqrt(npix*noise_sigma^2 + flux/exptime_s) for flux
    b = -snr**2 / exptime_s
    c = -snr**2 * npix 
    flux = (-b + np.sqrt(b**2 - 4*c)) / 2

    signal = psf * flux

    background_noise = np.random.normal(0, noise_sigma, (image_size, image_size))
    poisson_std = np.sqrt(np.clip(signal, 0, None) / exptime_s)
    poisson_noise = np.random.normal(0, poisson_std)

    frame = signal + background_noise + poisson_noise

    PSF_fitter = PSF_Fitter(image_size, prf, central_shape=5, function='sq')
    PSF_fitter.fit_psf(frame, 0.5, 0.5)

    return [PSF_fitter.source_x, PSF_fitter.source_y]

    # centx_s = image_size//2 + shift[0]
    # centy_s = image_size//2 + shift[1]

    # psf = prf.locate(centx_s,centy_s, (image_size,image_size))
    # psf /= np.nansum(psf[image_size//2-1:image_size//2+2,image_size//2-1:image_size//2+2])

    # noise_frame = np.random.normal(0,noise_sigma,(image_size,image_size))
    
    # flux = snr * (np.sqrt(npix)*noise_sigma)
    
    # signal = psf * flux
    # frame = signal + noise_frame

    # PSF_fitter = PSF_Fitter(image_size,prf,central_shape=5,function='sq')
    # PSF_fitter.fit_psf(frame,0.5,0.5)

    # return [PSF_fitter.source_x,PSF_fitter.source_y]

def simulate_cut_psf_fitting(path,sector,cam,ccd,cut,n=8,nfits=1000,nMedians=20,image_size=7,plot=False,return_models=False):

    from .dataprocessor import DataProcessor
    from PRF import TESS_PRF
    datadir='/fred/oz335/_local_TESS_PRFs/'
    from joblib import Parallel, delayed 
    from tqdm import tqdm
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import pickle


    dp = DataProcessor(sector=sector,data_path='/fred/oz335/TESSdata')
    _, cutCentrePx, _, _ = dp.find_cuts(cam=cam,ccd=ccd,n=n,plot=False)

    if sector>3:
        prf = TESS_PRF(sector=sector,cam=cam,ccd=ccd,colnum=cutCentrePx[cut-1][0],rownum=cutCentrePx[cut-1][1],localdatadir=datadir+'Sectors4+')
    else:
        prf = TESS_PRF(sector=sector,cam=cam,ccd=ccd,colnum=cutCentrePx[cut-1][0],rownum=cutCentrePx[cut-1][1],localdatadir=datadir+'Sectors1_2_3')


    snrs = 10 ** np.random.uniform(np.log10(1), np.log10(100), nfits)
    shifts = np.random.uniform(-0.5,0.5,(nfits,2))

    fits = Parallel(n_jobs=-1)(delayed(gen_and_fit_source)(snrs[i],shifts[i],image_size,prf) for i in tqdm(range(nfits),desc='    Fitting injected sources'))

    fits = np.array(fits)
    dx = fits[:,0] - shifts[:,0]
    dy = fits[:,1] - shifts[:,1]

    edges = np.logspace(np.log10(snrs.min()), np.log10(snrs.max()), nMedians + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # per-axis 68th percentiles, computed separately - matches get_wcs_uncertainty's approach
    x68 = np.array([
        np.percentile(np.abs(dx[(snrs >= edges[i]) & (snrs < edges[i+1])]), 68)
        for i in range(nMedians)
    ])
    y68 = np.array([
        np.percentile(np.abs(dy[(snrs >= edges[i]) & (snrs < edges[i+1])]), 68)
        for i in range(nMedians)
    ])

    # fit each separately vs SNR
    p0x = [np.log10(x68[0]), 0.7, np.log10(x68[-1])]
    poptx_log, _ = curve_fit(log_model_quadrature, centers, np.log10(x68), p0=p0x, maxfev=20000,
                          bounds=([-3, 0, -3], [1, 5, 1]))
    popty = 10**poptx_log[0], poptx_log[1], 10**poptx_log[2]

    p0y = [np.log10(y68[0]), 0.7, np.log10(y68[-1])]
    popty_log, _ = curve_fit(log_model_quadrature, centers, np.log10(y68), p0=p0y, maxfev=20000,
                            bounds=([-3, 0, -3], [1, 5, 1]))
    poptx = 10**popty_log[0], popty_log[1], 10**popty_log[2]



    # p0x = [np.max(x68), 0.5, np.min(x68)]
    # poptx, _ = curve_fit(model, centers, x68, p0=p0x, maxfev=10000)

    # p0y = [np.max(y68), 0.5, np.min(y68)]
    # popty, _ = curve_fit(model, centers, y68, p0=p0y, maxfev=10000)

    
    # if plot:
    #     snr_space = np.logspace(0,2,1000)
    #     r68_fit = model(snr_space,popt[0],popt[1],popt[2])

    #     plt.figure()
    #     plt.scatter(snrs,diffs,s=1)
    #     plt.plot(centers,medians,'x',c='r')
    #     plt.plot(centers,r68,'x',c='green')

    #     plt.plot(snr_space,med_fit,c='r',alpha=0.8)
    #     plt.plot(snr_space,r68_fit,c='g',alpha=0.8)

    #     plt.xscale('log')

    df = pd.DataFrame()
    df['snr'] = snrs
    df['dx'] = dx
    df['dy'] = dy

    np.save(f'{path}/cam{cam}_ccd{ccd}/snr_to_localisation/cut{cut}of{int(n**2)}_coeffs_x.npy',poptx)
    np.save(f'{path}/cam{cam}_ccd{ccd}/snr_to_localisation/cut{cut}of{int(n**2)}_coeffs_y.npy',popty)
    df.to_csv(f'{path}/cam{cam}_ccd{ccd}/snr_to_localisation/cut{cut}of{int(n**2)}_sources.csv',index=False)

    # with open(f'{path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{int(n**2)}/snr_localisation_coeffs_x.pkl', 'wb') as file:
    #     pickle.dump(poptx, file)
    # with open(f'{path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{int(n**2)}/snr_localisation_coeffs_y.pkl', 'wb') as file:
    #     pickle.dump(popty, file)
    print('    SNR to localisation accuracy model generated')

    if return_models:
        return get_snr_to_localisation_func(path,sector,cam,ccd,cut)


def get_snr_to_localisation_func(path,sector,cam,ccd,cut,n=8,xy=True):

    import pickle

    if xy:

        poptx = np.load(f'{path}/cam{cam}_ccd{ccd}/snr_to_localisation/cut{cut}of{int(n**2)}_coeffs_x.npy')
        popty = np.load(f'{path}/cam{cam}_ccd{ccd}/snr_to_localisation/cut{cut}of{int(n**2)}_coeffs_y.npy')

        def func_x(snr):
            return model_quadrature(snr, *poptx) 
        def func_y(snr):
            return model_quadrature(snr, *popty) 
        
        return func_x,func_y

    else:
        with open(f'{path}/Sector{sector}/Cam{cam}/Ccd{ccd}/Cut{cut}of{int(n**2)}/snr_localisation_coeffs.pkl', 'rb') as file:
            popt = pickle.load(file)
        
        def func(snr):
            return model(snr, *popt)

        return func

def get_wcs_uncertainty(path,sector,cam,ccd,cut,n=8):

    ccd_sources = pd.read_csv(f'{path}/Sector{sector}/Cam{cam}/Ccd{ccd}/wcs/ref/ccd_sourcefits.csv')
    cut_wcs = CutWCS(path,sector,cam,ccd,cut,n)

    corner = cut_wcs.corner
    size = 2048/n

    cut_sources = ccd_sources[(ccd_sources.xPSF>corner[0])&(ccd_sources.xPSF<corner[0]+size)&
                              (ccd_sources.yPSF>corner[1])&(ccd_sources.yPSF<corner[1]+size)]

    x,y = cut_wcs.all_world2pix(cut_sources.ra,cut_sources.dec,0)

    dx = cut_sources.xPSF - (x+corner[0])
    dy = cut_sources.yPSF - (y+corner[1])
    xstd = np.percentile(np.abs(dx),68)
    ystd = np.percentile(np.abs(dy),68)

    return xstd, ystd