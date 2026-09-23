import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval, AsinhStretch

import requests
from PIL import Image
from io import BytesIO
from time import sleep 

import pandas as pd
from matplotlib.patches import Ellipse
from skimage.transform import rotate
from astropy.stats import sigma_clipped_stats
import os


fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def _Fetch(url, timeout=60):

    """
    Download with requests rather than giving astropy the URL, which prints a
    'Downloading ...' progress bar and keeps a copy in ~/.astropy/cache.
    """

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response

def _Get_images(ra,dec,filters):

    """Query ps1filenames.py service to get a list of images"""

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(_Fetch(url).text, format='ascii')
    return table

def _Get_url_wcs(ra, dec, size, filters, color=False):

    """Get URL for images in the table"""

    table = _Get_images(ra,dec,filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size={size}&format=fits")

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)

    return url

def _Get_im(ra, dec, size,color):

    """Get color image at a sky position"""

    if color:
        url = _Get_url_wcs(ra,dec,size=size,filters='grz',color=True)
        fh = fits.open(BytesIO(_Fetch(url).content))
        wcs = WCS(fh[0])
        # r = requests.get(url)
        # im = Image.open(BytesIO(r.content))

        fim = fh[0].data

        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)

        # transform = AsinhStretch() + PercentileInterval(99.5)
        # bfim = transform(im)
        im = bfim
        
    else:
        url = _Get_url_wcs(ra,dec,size=size,filters='i')
        fh = fits.open(BytesIO(_Fetch(url[0]).content))
        wcs = WCS(fh[0])

        fim = fh[0].data
        # replace NaN values with zero for display
        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)
        im = bfim

    return im,wcs

# def _Panstarrs_phot(ra,dec,size):

#     grey_im,wcsI = _Get_im(ra,dec,size=size*6,color=False)
#     colour_im,wcsGRZ = _Get_im(ra,dec,size=size*6,color=True)

#     wcsList = [wcsGRZ,wcsGRZ,wcsGRZ,wcsI]

#     plt.rcParams.update({'font.size':12})
#     fig,ax = plt.subplots(ncols=4,figsize=(3*fig_width,1*fig_width))

#     ax[2].imshow(grey_im,origin="lower",cmap="gray")
#     ax[2].set_title('PS1 i')
#     ax[2].set_xlabel('px (0.25")')
#     ax[3].imshow(colour_im[0],origin="lower",cmap="gray")
#     ax[3].set_title('PS1 z')
#     ax[3].set_xlabel('px (0.25")')
#     ax[1].imshow(colour_im[1],origin="lower",cmap="gray")
#     ax[1].set_title('PS1 r')
#     ax[1].set_xlabel('px (0.25")')
   
#     ax[0].set_title('PS1 g')
#     ax[0].imshow(colour_im[2],origin="lower",cmap="gray")
#     ax[0].set_xlabel('px (0.25")')

#     return fig,wcsList,grey_im.shape[0]

def _panstarrs_objects(ra, dec, rad=80, release='dr2'):
    
    # convert radius to degrees
    rad_deg = rad / 3600.0
    
    # Build MAST Pan-STARRS API URL
    url = (
        f"https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/{release}/"
        f"stack.csv?ra={ra}&dec={dec}&radius={rad_deg}&nDetections.gte=1"
    )
    
    try:
        ps = pd.read_csv(url)
    except Exception as e:
        print(f"Query failed: {e}")
        return None
    
    if len(ps) > 0:
        # Keep useful columns
        keep = [
            'objID','raMean','decMean','gPSFMag','gKronMag','rPSFMag','rKronMag',
            'iPSFMag','iKronMag','zPSFMag','zKronMag','yPSFMag','yKronMag',
            'gPSFMagErr','gKronMagErr','rPSFMagErr','rKronMagErr','iPSFMagErr',
            'iKronMagErr','zPSFMagErr','zKronMagErr','yPSFMagErr','yKronMagErr',
            'qualityFlag','primaryDetection'
        ]
        ps = ps[keep]
        ps = ps.rename(columns={
            'raMean':'ra','decMean':'dec',
        })
        
        # crude star/galaxy separation using PS1 "primaryDetection" and qualityFlag
        ps['star'] = 0
        ps['star'].loc[ps['iPSFMag']-ps['iKronMag'] < 0.1] = 2
        ps['star'].loc[ps['iPSFMag']-ps['iKronMag'] < 0.05] = 1
    else:
        ps = None

    
    return ps

def _Panstarrs_phot(ra,dec,size):

    # grey_im,wcsI = _Get_im(ra,dec,size=size*6,color=False)
    colour_im,wcsGRZ = _Get_im(ra,dec,size=size*6,color=True)

    wcs = wcsGRZ.dropaxis(2)

    truegrz = []
    for im in colour_im[::-1]:
        m,med,std = sigma_clipped_stats(im)
        im -= med
        pixels = im.flatten()
        p25, p75 = np.percentile(pixels, 15), np.percentile(pixels, 85)
        central_pixels = pixels[(pixels >= p25) & (pixels <= p75)]
        std_central = np.std(central_pixels)
        im = im / std_central
        im[im<0]=0
        truegrz.append(im)
    
    truegrz[1] = (truegrz[2]+truegrz[0])/2
    rgb = np.dstack([truegrz[2],truegrz[1],truegrz[0]])
    m,med,std = sigma_clipped_stats(rgb)
    rgb = rgb / (med+10*std)
    rgb = _Stretch_rgb(rgb, stretch=1, gamma=1)
    rgb /= np.nanmax(rgb)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection=wcs)
    ax.imshow(rgb, origin='lower')
    ax.set_xlabel('Right Ascension',fontsize=17)
    ax.set_ylabel('Declination',fontsize=17)
    ax.set_title('PanSTARRS gz',fontsize=17)
    ax.invert_xaxis()
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm:ss')

    return fig,wcs,truegrz[0].shape[0],rgb


def _skymapper_objects(ra,dec,imshape,wcs,rad=60):
    """
    radius in degrees
    """
    rad /= 60**2 

    query = f'https://skymapper.anu.edu.au/sm-cone/public/query?RA={ra}&DEC={dec}&SR={np.round(rad,3)}&RESPONSEFORMAT=CSV'
    sm = pd.read_csv(query)
    if len(sm) > 0:
        keep = ['object_id','raj2000','dej2000','u_psf', 'e_u_psf',
                'v_psf', 'e_v_psf','g_psf', 'e_g_psf','r_psf', 
                'e_r_psf','i_psf', 'e_i_psf','z_psf', 'e_z_psf','class_star']
        sm = sm[keep]
        sm = sm.rename(columns={'raj2000':'ra','dej2000':'dec','u_psf':'umag',
                                'v_psf':'vmag','g_psf':'gmag','r_psf':'rmag',
                                'i_psf':'imag','z_psf':'zmag',
                                'e_u_psf':'e_umag','e_v_psf':'e_vmag','e_g_psf':'e_gmag',
                                'e_r_psf':'e_rmag','e_i_psf':'e_imag','e_z_psf':'e_zmag'})
        sm['star'] = 0
        sm['star'].loc[sm['class_star'] >= 0.9] = 1
        sm['star'].loc[sm['class_star'] <= 0.7] = 0
        sm['star'].loc[(sm['class_star'] > 0.7) & (sm['class_star'] < 0.9)] = 2
    else:
        sm = None

    # x,y = wcs.all_world2pix(sm['ra'],sm['dec'],0)
    # ydiff = y - imshape//2
    # y = imshape//2 - ydiff
    # ra,dec = wcs.all_pix2world(x,y,0)
    # sm['ra'] = ra
    # sm['dec'] = dec

    return sm

# Step 3: apply stretch to the whole RGB image
def _Stretch_rgb(rgb_img, stretch=5, gamma=0.45):
    # Apply arcsinh stretch per channel but keep ratios
    stretched = np.arcsinh(stretch * rgb_img) / np.arcsinh(stretch)
    stretched = np.clip(stretched ** gamma, 0, 1)
    return stretched


def _Adjust_band_peaks(rgb, band_idx, sigma_thresh=3, scale_factor=0.7,exponent=5):

    rgb_out = rgb.copy()
    band = rgb_out[:, :, band_idx]
    # Flatten and get central 50% pixels for noise std estimation
    flat = band.flatten()
    q25, q75 = np.percentile(flat, [25, 75])
    central_pixels = flat[(flat >= q25) & (flat <= q75)]
    noise_std = np.nanstd(central_pixels)
    
    threshold = sigma_thresh * noise_std
    
    mask = band > threshold
    if np.any(mask):
        min_val = threshold
        max_val = np.max(band[mask])
        normalized = (band[mask] - min_val) / (max_val - min_val)  # 0 to 1
        # Nonlinear scaling: scale more for moderate pixels, less for extreme pixels
        scaling = scale_factor + (1 - scale_factor) * (normalized ** exponent)
        band[mask] = band[mask] * scaling
    
    rgb_out[:, :, band_idx] = band
    return rgb_out

def _Rotate_i_to_g(i_image, wcs_i, g_image, wcs_g, n_points=5):

    ny, nx = i_image.shape
    
    x = np.linspace(0, nx-1, n_points)
    y = np.linspace(0, ny-1, n_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    
    world_coords = wcs_i.pixel_to_world(xv, yv)
    ra = world_coords.ra.deg
    dec = world_coords.dec.deg
    
    g_x, g_y = wcs_g.world_to_pixel(world_coords)
    
    cx_i, cy_i = (nx-1)/2, (ny-1)/2
    cx_g, cy_g = (g_image.shape[1]-1)/2, (g_image.shape[0]-1)/2
    
    dx_i = xv - cx_i
    dy_i = yv - cy_i
    dx_g = g_x - cx_g
    dy_g = g_y - cy_g
    
    angles = np.arctan2(dy_g, dx_g) - np.arctan2(dy_i, dx_i)
    angle_deg = np.median(np.degrees(angles))
    
    rotated_i = rotate(i_image, angle=-angle_deg, resize=False, center=(cx_i, cy_i))
    
    return rotated_i, angle_deg


def _Skymapper_phot(ra, dec, size, show_bands=False,verbose=False):
    """
    Gets g, r, i from SkyMapper and makes a colour composite:
    g -> blue, r -> green, i -> red
    
    If show_bands=True, plots the three bands separately.
    """

    from astropy import log
    log.setLevel('ERROR')

    size *= 1.5
    og_size = size
    size_deg = size / 3600  # arcsec → degrees

    url = f"https://api.skymapper.nci.org.au/public/siap/dr4/query?POS={ra},{dec}&SIZE={size_deg}&BAND=g,r,i&FORMAT=GRAPHIC&VERB=3&INTERSECT=COVERS"
    max_attempts = 5

    complete = False
    attempt = 0
    while (not complete) & (attempt < max_attempts):
        try:
            table = Table.read(_Fetch(url).text, format='ascii').to_pandas()
            complete = True
        except:
            attempt += 1
            if verbose:
                print('failed to get table')
            sleep(1)

    t = table[(table['col16'] == 'main')]
    t_unique = t[~t.duplicated(subset='col3', keep='first')]

    wcsList = []
    images = []
    filts = ['g', 'r', 'i']
    for filt in filts:
        complete = False
        attempt = 0
        while (not complete) & (attempt < max_attempts):
            try:
                f = t_unique.loc[t_unique['col3'] == filt].iloc[0]
                img_url = f['col6']
                hdu = fits.open(BytesIO(_Fetch(img_url).content))
                data = hdu[0].data
                m,med,std = sigma_clipped_stats(data)
                data -= med
                im = data * 10**((hdu[0].header['ZPAPPROX'] - 25)/-2.5)

                im[im == 0] = np.nan
                im -= np.nanmedian(im)
                im[np.isnan(im)] = 0

                wcs = WCS(hdu[0].header)

                pixels = im.flatten()
                p25, p75 = np.percentile(pixels, 15), np.percentile(pixels, 85)
                central_pixels = pixels[(pixels >= p25) & (pixels <= p75)]
                std_central = np.std(central_pixels)
                im = im / std_central
                im[im<0]=0
                images.append(im)

                wcsList.append(wcs)
                complete = True
            except:
                attempt += 1
                sleep(1)

    if len(images)<3:
        return None,None,None,None

    rotated_i, angle = _Rotate_i_to_g(images[2], wcsList[2], images[0],wcsList[0])
    images[2] = rotated_i

    min_shape = np.min([im.shape for im in images], axis=0)
    cropped_images = [im[:min_shape[0], :min_shape[1]] for im in images]

    rgb = np.dstack([cropped_images[2], cropped_images[1], cropped_images[0]])

    m,med,std = sigma_clipped_stats(data)

    rgb = rgb / (med+1*std)
    # rgb_norm = rgb / np.nanmax(rgb)
    rgb = _Stretch_rgb(rgb, stretch=1, gamma=0.7)

    rgb /= np.nanmax(rgb)

    plt.rcParams.update({'font.size': 12})
    if show_bands:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5),subplot_kw={'projection':wcsList[0]})
        fig.suptitle('SkyMapper gri',fontsize=17)
        bands = ['g (blue)', 'r (green)', 'i (red)']
        for i, ax in enumerate(axs):
            if i ==0 :
                ax.set_xlabel('Right Ascension',fontsize=17)
                ax.set_ylabel('Declination',fontsize=17)
                ax.coords[0].set_major_formatter('hh:mm:ss')
                ax.coords[1].set_major_formatter('dd:mm:ss')
            else:
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[1].set_ticklabel_visible(False)
            im = rgb[:,:,i]
            ax.imshow(im, cmap='gray', origin='lower',vmin=0,vmax=1)
            ax.set_title(f"{bands[i]} band")


    else:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection=wcsList[0])
        ax.imshow(rgb, origin='lower')
        ax.set_xlabel('Right Ascension',fontsize=17)
        ax.set_ylabel('Declination',fontsize=17)
        ax.set_title('SkyMapper gri',fontsize=17)
        ax.invert_xaxis()
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')

    return fig, wcsList[0], rgb.shape[0],rgb

def _delve_objects(ra,dec,size=60/60**2):
    from dl import queryClient as qc
    query = f"""
        SELECT o.quick_object_id,o.ra, o.dec,
        o.mag_psf_g,o.mag_psf_r,o.mag_psf_i,o.mag_psf_z,
        o.mag_auto_g,o.mag_auto_r,o.mag_auto_i,o.mag_auto_z,
        o.magerr_psf_g,o.magerr_psf_r,o.magerr_psf_i,o.magerr_psf_z,
        o.magerr_auto_g,o.magerr_auto_r,o.magerr_auto_i,o.magerr_auto_z,
        o.extended_class_g, o.extended_class_r,o.extended_class_i,o.extended_class_z
        FROM delve_dr2.objects AS o
        WHERE q3c_radial_query(ra,dec,{ra},{dec},{size})
        """
    try:
        result = qc.query(sql=query,fmt='pandas')
        result.replace(99.0,np.nan,inplace=True)
        result['star'] = 0
        ty = []
        val = result[['extended_class_g','extended_class_r','extended_class_i','extended_class_z']].values
        for v in val:
            if 3 in v:
                ty += [0]
            elif 1 in v:
                ty += [1]
            else:
                ty += [2]
        result['star'] = ty
        return result
    except:
        return

def _DESI_phot(ra,dec,size,verbose):

    size = size * 5
    urlFITS = f"http://legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&size={size}&layer=ls-dr10"
    urlIM = f"http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&size={size}&layer=ls-dr10"

    response = requests.get(urlIM)
    if response.status_code == 200:
        image = Image.open(BytesIO(requests.get(urlIM).content))
        #if np.nansum(image) == 0:
        try:
            hdulist = fits.open(BytesIO(requests.get(urlFITS).content),ignore_missing_simple=True)
            hdu = hdulist[0]
            wcs = WCS(hdu.header)
            wcs = wcs.dropaxis(2)

            plt.rcParams.update({'font.size':12})
            fig = plt.figure(figsize=(8,8))#3*fig_width,1*fig_width))
            ax = plt.subplot(111,projection=wcs)
            ax.imshow(image,cmap="gray")
            ax.set_title('DES gri',fontsize=17)
            ax.grid(alpha=0.2)
            ax.set_xlabel('Right Ascension',fontsize=17)
            ax.set_ylabel('Declination',fontsize=17)
            ax.invert_xaxis()
            return fig,wcs,size,image
        except Exception as error:
            if verbose:
                print("DES Photometry failed: ", error)
                print(urlFITS)
                print(urlIM)
            return None,None,None,None
        #else:
          #  return None,None,None
    else:
        return None,None,None,None

def simbad_sources(ra,dec,size):
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.ROW_LIMIT = -1
    gal_query = f"""SELECT ra, dec, main_id, otype

                    FROM basic

                    WHERE otype != 'star..'

                    AND CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra}, {dec} , {size})) = 1
                """
    gal = simbad.query_tap(gal_query)
    gal = gal.to_pandas()

    star_query = f"""SELECT ra, dec, main_id, otype

                    FROM basic

                    WHERE otype = 'star..'

                    AND CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra}, {dec} , {size})) = 1
                 """
    stars = simbad.query_tap(star_query)
    stars = stars.to_pandas()

    gal['star'] = 0
    stars['star'] = 1
    sbad = pd.concat([stars,gal])
    
    return sbad

def check_simbad(cat,sbad):
    dist = np.sqrt((sbad['ra'].values[:,np.newaxis] - cat['ra'].values[np.newaxis,:])**2 + (sbad['dec'].values[:,np.newaxis] - cat['dec'].values[np.newaxis,:])**2)*60**2
    sind, catind = np.where(dist<5)
    cat['otype'] = 'none'
    if len(catind) > 0:
        cat.loc[catind,'star'] = sbad.loc[sind,'star'].values
        cat.loc[catind,'otype'] = sbad.loc[sind,'otype'].values
    return cat

def get_gaia(ra, dec, size):
    from astroquery.gaia import Gaia, GaiaClass
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    Gaia.ROW_LIMIT = -1
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    try:
        Gaia.login(user='hroxburg', password='ga!aRhoxu135')
        j = Gaia.cone_search_async(coord, radius=u.Quantity(100, u.arcsec))
        j = j.get_results()
    except Exception as e:
        print(f"Async query failed ({e}), falling back to anonymous synchronous...")
        try:
            Gaia.logout()
        except Exception:
            pass
        # Reinitialise entirely to clear broken session
        Gaia = GaiaClass()
        Gaia.ROW_LIMIT = -1
        j = Gaia.cone_search(coord, radius=u.Quantity(100, u.arcsec))
        j = j.get_results()

    j = j.to_pandas()
    j['star'] = 1
    j.loc[(j['classprob_dsc_combmod_quasar'] > 0.7) | (j['classprob_dsc_combmod_galaxy'] > 0.7), 'star'] = 2
    j.loc[(j['classprob_dsc_combmod_quasar'] > 0.9) | (j['classprob_dsc_combmod_galaxy'] > 0.9), 'star'] = 0
    return j

def get_gaia_loc(ra,dec,size,path):
    import pandas as pd

    gaia = pd.read_csv(path)
    gaia = gaia[(abs(gaia.ra-ra)<size)&(abs(gaia.dec-dec)<size)]
    return gaia


def get_gaia_vizier(ra, dec, size):
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag', 
                        'PGal', 'PQso'], 
               row_limit=-1)
    coord = SkyCoord(ra=ra, dec=dec, unit=u.degree)
    result = v.query_region(coord, 
                            radius=100*u.arcsec, 
                            catalog='I/355/gaiadr3')
    if result:
        j = result[0].to_pandas()
        j['star'] = 1
        j.loc[(j['PQso'] > 0.7) | (j['PGal'] > 0.7), 'star'] = 2
        j.loc[(j['PQso'] > 0.9) | (j['PGal'] > 0.9), 'star'] = 0
        return j

def check_gaia(cat,gaia):
    dist = np.sqrt((gaia['ra'].values[:,np.newaxis] - cat['ra'].values[np.newaxis,:])**2 + (gaia['dec'].values[:,np.newaxis] - cat['dec'].values[np.newaxis,:])**2)*60**2
    gind, catind = np.where(dist<5)
    cat['dist'] = np.nan
    cat['dist_l'] = np.nan
    cat['dist_u'] = np.nan
    if len(catind) > 0:
        cat.loc[catind,'star'] = gaia.loc[gind,'star'].values
        cat.loc[catind,'dist'] = gaia.loc[gind,'distance_gspphot'].values
        cat.loc[catind,'dist_l'] = gaia.loc[gind,'distance_gspphot_lower'].values
        cat.loc[catind,'dist_u'] = gaia.loc[gind,'distance_gspphot_upper'].values
        
    return cat

def check_local_gaia(cat,gaia):
    dist = np.sqrt((gaia['ra'].values[:,np.newaxis] - cat['ra'].values[np.newaxis,:])**2 + (gaia['dec'].values[:,np.newaxis] - cat['dec'].values[np.newaxis,:])**2)*60**2
    gind, catind = np.where(dist<5)
    # cat['dist'] = np.nan
    # cat['dist_l'] = np.nan
    # cat['dist_u'] = np.nan
    if len(catind) > 0:
        cat.loc[catind,'star'] = 1
        # cat.loc[catind,'dist'] = gaia.loc[gind,'distance_gspphot'].values
        # cat.loc[catind,'dist_l'] = gaia.loc[gind,'distance_gspphot_lower'].values
        # cat.loc[catind,'dist_u'] = gaia.loc[gind,'distance_gspphot_upper'].values
        
    return cat
    
    

def _add_sources(fig,cat):
    axs = fig.get_axes()
    count = 0
    for ax in axs:

        # scatter stars 
        stars = cat.loc[cat['star'] ==1]
        ax.scatter(stars.ra,stars.dec, transform=ax.get_transform('fk5'),
                    edgecolors='bisque',marker='*',s=80,facecolors="None",linewidths=1,label='Star')

        # scatter galaxies
        gal = cat.loc[cat['star'] == 0]
        ax.scatter(gal.ra,gal.dec, transform=ax.get_transform('fk5'),
                    edgecolors='bisque',marker='o',s=80,facecolors="None",label='Galaxy')
        
        # scatter other
        gal = cat.loc[cat['star'] == 2]
        ax.scatter(gal.ra,gal.dec, transform=ax.get_transform('fk5'),
                    edgecolors='bisque',marker='D',s=80,facecolors="None",label='Possible galaxy')
        # if count == 0:
        #     legend = ax.legend(loc=2,facecolor="black",fontsize=10)
        #     for text in legend.get_texts():
        #         text.set_color("white")
        count += 1
    return fig


def event_cutout(coords,size=50,phot=None,check='gaia',verbose=True):
    """
    Make an image using ground catalogs for the region of interest.

    parameters
    ----------
    coords : array
        array with elements of (ra,dec) in decimal degrees
    real_loc : array
        real on-sky position of target, if known.
    error : float
        Positional error in arcseconds, currently only 1 value is accepted.
    size : int
        size of the cutout image, likely in arcseconds 
    phot : str
        String defining the catalog to use. If None, then the catalog is automatically determined.
    check : str
        Check the photometry catalog against another catalog with better star detection. Currenty
        only gaia and simbad are available.
    """

    if phot is None:
        fig,wcs,outsize,im = _DESI_phot(coords[0],coords[1],size,verbose)
        if fig is None:
            if coords[1] > -28:
                phot = 'PS1'
            else:
                phot = 'SkyMapper'
        else:
            phot = 'DESI'
            cat = _delve_objects(coords[0],coords[1])

    if phot == 'PS1':
        fig,wcs,outsize,im = _Panstarrs_phot(coords[0],coords[1],size)
        if fig is None:
            if verbose:
                print('PanSTARRS photometry failed.')
            return None,None,None,None,None,None
        cat = _panstarrs_objects(coords[0],coords[1])
        #fig = _add_sources(fig,cat)

    elif phot.lower() == 'skymapper':
        fig,wcs,outsize,im = _Skymapper_phot(coords[0],coords[1],size,verbose=verbose)
        if fig is None:
            if verbose:
                print('SkyMapper photometry failed.')
            return None,None,None,None,None,None
        cat = _skymapper_objects(coords[0],coords[1],im.shape[1],wcs,rad=60)
    elif phot is None:
        if verbose:
                print('Photometry name invalid.')
        fig = None
        wcs = None
        return None,None,None,None,None,None
        
    # if phot is not None:
    if check == 'simbad':
        sbad = simbad_sources(coords[0],coords[1],size/60**2)
        cat = check_simbad(cat,sbad)
    elif check == 'gaia':
        gaia = get_gaia(coords[0],coords[1],size/60**2)
        if (gaia is not None) & (cat is not None):
            cat = check_gaia(cat,gaia)
        elif phot == 'SkyMapper':
            if verbose:
                print('Something failed getting Skymapper sources.')
            return None,None,None,None,None,None
    elif os.path.exists(check):
        gaia = get_gaia_loc(coords[0],coords[1],size/60**2,check)
        cat = check_local_gaia(cat,gaia)

    if cat is not None:
        fig = _add_sources(fig,cat)

    plt.close()

    return fig,wcs,outsize, phot, cat,im