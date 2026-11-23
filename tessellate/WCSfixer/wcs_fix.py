from .pdastro import *
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pysiaf
import pandas as pd
import gc 
from astropy.io import fits
from copy import deepcopy
from astropy.wcs import WCS

def polyfit0(u, x, y, order,fit_coeffs0=False,rotation_fit_x=True, rotation_fit_y=True, weight=None):
    """Fit polynomial to a set of u values on an x,y grid.
    u is a function u(x,y) being a polynomial of the form
    u = a[i, j] x**(i-j) y**j. x and y can be on a grid or be arbitrary values
    This version uses scipy.linalg.solve instead of matrix inversion.
    u, x and y must have the same shape and may be 2D grids of values.
    
    **** NOTE: in this routine coeffs[0] is kept at 0 if fit_coeffs0=False! ****
    
    Parameters
    ----------
    u : array
        an array of values to be the results of applying the sought after
        polynomial to the values (x,y)
    x : array
        an array of x values
    y : array
        an array of y values
    order : int
        the polynomial order
    Returns
    -------
    coeffs: array
        polynomial coefficients being the solution to the fit.
    """
    # First set up x and y powers for each coefficient
    px = []
    py = []

    if rotation_fit_x == False or rotation_fit_y == False:
        startindex = 2
    elif fit_coeffs0 == False:
        ### Do not fit coeffs[0]!!!!
        startindex = 1
    else:
        startindex = 0
        
    for i in range(startindex,order + 1):
        for j in range(i + 1):
            px.append(i - j)
            py.append(j)
    terms = len(px)
    
    if rotation_fit_x == False:
        u = u - x
    elif rotation_fit_y == False:
        u = u - y

    # Make up matrix and vector
    vector = np.zeros((terms))
    mat = np.zeros((terms, terms))
    for i in range(terms):
        vector[i] = (u * x ** px[i] * y ** py[i]).sum()  # Summing over all x,y
        for j in range(terms):
            mat[i, j] = (x ** px[i] * y ** py[i] * x ** px[j] * y ** py[j]).sum()

    coeffs = linalg.solve(mat, vector)
    
    if rotation_fit_x == False:
        coeffs0 = [0.0, 1.0, 0.0]
        coeffs0.extend(coeffs)
    elif rotation_fit_y == False:
        coeffs0 = [0.0, 0.0, 1.0]
        coeffs0.extend(coeffs)
    elif fit_coeffs0 == False:
        ### Add coeffs[0]=0.0
        coeffs0 = [0.0]
        coeffs0.extend(coeffs)
    else:
        coeffs0 = coeffs
    
    return coeffs0

def plot_residuals(table, ixs_good, ixs_cut, ixs_excluded, sp=None, residual_limits=(-0.4,0.4),
                  xfigsize4subplot=9, yfigsize4subplot=3,
                  add2title=None):
    
    if sp is None:
        sp=initplot(4,1, xfigsize4subplot=xfigsize4subplot, yfigsize4subplot=yfigsize4subplot)

    plot_style={}
    plot_style['good']={'style':'o','color':'blue', 'ms':5 ,'alpha':0.5}
    plot_style['cut']={'style':'o','color':'red', 'ms':5 ,'alpha':0.3}
    plot_style['excluded']={'style':'o','color':'gray', 'ms':3 ,'alpha':0.3}

    # beware: these are global variables!
    title = f'CCD Fits'
    
    if add2title is not None: title+=f' {add2title}'
    ### dy vs y
    if len(ixs_excluded)>0: table.t.loc[ixs_excluded].plot('yPSF','dv_pix',ax=sp[0],ylim=residual_limits,ylabel='dv_pix', **plot_style['excluded'])
    if len(ixs_cut)>0: table.t.loc[ixs_cut].plot('yPSF','dv_pix',ax=sp[0],ylim=residual_limits,ylabel='dv_pix', **plot_style['cut'])
    table.t.loc[ixs_good].plot('yPSF','dv_pix',ax=sp[0],ylim=residual_limits,ylabel='dv_pix',title=title, **plot_style['good'])

    ### dx vs x
    if len(ixs_excluded)>0: table.t.loc[ixs_excluded].plot('xPSF','du_pix',ax=sp[1],ylim=residual_limits,ylabel='du_pix', **plot_style['excluded'])
    if len(ixs_cut)>0: table.t.loc[ixs_cut].plot('xPSF','du_pix',ax=sp[1],ylim=residual_limits,ylabel='du_pix', **plot_style['cut'])
    table.t.loc[ixs_good].plot('xPSF','du_pix',ax=sp[1],ylim=residual_limits,ylabel='du_pix', **plot_style['good'])

    ### dy vs x
    if len(ixs_excluded)>0: table.t.loc[ixs_excluded].plot('xPSF','dv_pix',ax=sp[2],ylim=residual_limits,ylabel='dv_pix', **plot_style['excluded'])
    if len(ixs_cut)>0: table.t.loc[ixs_cut].plot('xPSF','dv_pix',ax=sp[2],ylim=residual_limits,ylabel='dv_pix', **plot_style['cut'])
    table.t.loc[ixs_good].plot('xPSF','dv_pix',ax=sp[2],ylim=residual_limits,ylabel='dv_pix', **plot_style['good'])


    ### dx vs y
    if len(ixs_excluded)>0: table.t.loc[ixs_excluded].plot('yPSF','dv_pix',ax=sp[3],ylim=residual_limits,ylabel='dv_pix', **plot_style['excluded'])
    if len(ixs_cut)>0: table.t.loc[ixs_cut].plot('yPSF','dv_pix',ax=sp[3],ylim=residual_limits,ylabel='dv_pix', **plot_style['cut'])
    table.t.loc[ixs_good].plot('yPSF','dv_pix',ax=sp[3],ylim=residual_limits,ylabel='dv_pix', **plot_style['good'])

    for i in range(4): 
        sp[i].axhline(0,  color='black',linestyle='-', linewidth=2.0)

    #for i in range(3): sp[i].get_legend().remove()
    for i in range(len(sp)): 
        if sp[i].get_legend() is not None:
            sp[i].get_legend().remove()

    plt.tight_layout()
    return(sp)

def fit_Idl2Sci(NAXIS1,NAXIS2,CRPIX1,CRPIX2,coeff_Sci2IdlX,coeff_Sci2IdlY,poly_degree, fit_coeffs0=False, savepng=None):
    nx, ny = (int(NAXIS1/16), int(NAXIS2/16))
    x = np.linspace(1, NAXIS1, nx)
    y = np.linspace(1, NAXIS2, ny)
    xgprime, ygprime = np.meshgrid(x-(CRPIX1-1), y-(CRPIX2-1))
    
    xg_idl = pysiaf.utils.polynomial.poly(coeff_Sci2IdlX, xgprime, ygprime, order=poly_degree)
    yg_idl = pysiaf.utils.polynomial.poly(coeff_Sci2IdlY, xgprime, ygprime, order=poly_degree)
    coeff_Idl2SciX = polyfit0(xgprime, xg_idl, yg_idl, order=poly_degree, fit_coeffs0=fit_coeffs0)
    coeff_Idl2SciY = polyfit0(ygprime, xg_idl, yg_idl, order=poly_degree, fit_coeffs0=fit_coeffs0)
    
    return(coeff_Idl2SciX,coeff_Idl2SciY)


def make_coeff_table(coeff_Sci2IdlX,coeff_Sci2IdlY,coeff_Idl2SciX,coeff_Idl2SciY,poly_degree):
    coeffs = pdastroclass(columns=['CAMERA','CCD','siaf_index','exponent_x','exponent_y','Sci2IdlX','Sci2IdlY','Idl2SciX','Idl2SciY'])
    for i in range(poly_degree+1):
        exp_x = i
        for j in range(0,i+1):
            siaf_index = i*10+j
            coeffs.newrow({'siaf_index':siaf_index,
                           'exponent_x':exp_x,
                           'exponent_y':j
                        })
            exp_x -= 1
    coeffs.t['Sci2IdlX']=coeff_Sci2IdlX
    coeffs.t['Sci2IdlY']=coeff_Sci2IdlY
    coeffs.t['Idl2SciX']=coeff_Idl2SciX
    coeffs.t['Idl2SciY']=coeff_Idl2SciY

    coeffs.default_formatters['Sci2IdlX']='{:.10e}'.format
    coeffs.default_formatters['Sci2IdlY']='{:.10e}'.format
    coeffs.default_formatters['Idl2SciX']='{:.10e}'.format
    coeffs.default_formatters['Idl2SciY']='{:.10e}'.format

    coeffs.write()

    return coeffs 


def Calculate_WCS(save_folder,order=6):

    wcs_path = f'{save_folder}/corrected.fits'
    # -- Drop failed star fits: polyfit0's pandas sums skip NaN, but NaN**0 = 1, so a NaN-xPSF star still enters
    # -- the constant-term equations -- one such star gave S98 Cam1 Ccd1 a 0.66 px ring-shaped WCS error
    ccd_sources = pd.read_csv(f'{save_folder}/ccd_sourcefits.csv')
    ccd_sources = ccd_sources.dropna(subset=['xPSF','yPSF','ra','dec']).reset_index(drop=True)

    imtable = pdastroclass()
    ix = imtable.newrow({'basename':f'tess_image'}) 
    imtable.t.loc[ix,'imID']=int(ix)

    image = pd.DataFrame()
    image['Source'] = ccd_sources.Source
    image['min_dist_arcsec'] = ccd_sources.min_dist_arcsec
    image['raGaia'] = ccd_sources.ra
    image['decGaia'] = ccd_sources.dec
    image['xPSF'] = ccd_sources.xPSF
    image['yPSF'] = ccd_sources.yPSF

    with fits.open(wcs_path, memmap=False) as hdul:
        header = hdul[1].header.copy()

    image['u'], image['v'] = radec_to_uvprime(image['raGaia'],image['decGaia'],header)

    ixs_im = imtable.getindices()

    imtable.t.loc[ixs_im[0],'CRPIX1']=header['CRPIX1']
    imtable.t.loc[ixs_im[0],'CRPIX2']=header['CRPIX2']
    imtable.t.loc[ixs_im[0],'NAXIS1']=header['NAXIS1']
    imtable.t.loc[ixs_im[0],'NAXIS2']=header['NAXIS2']
    image['xprime'] = image['xPSF']-(imtable.t.loc[ixs_im[0],'CRPIX1'] - 1)
    image['yprime'] = image['yPSF']-(imtable.t.loc[ixs_im[0],'CRPIX2'] - 1)
    image['imID'] = imtable.t.loc[ixs_im[0],'imID']


    # error checking: is there a global CRPIX1/2?
    CRPIX1 = unique(imtable.t['CRPIX1'])[0]
    CRPIX2 = unique(imtable.t['CRPIX2'])[0]

    # error checking: is there a global NAXIS1/2?
    NAXIS1 = unique(imtable.t['NAXIS1'])[0]
    NAXIS2 = unique(imtable.t['NAXIS2'])[0]

    phottablelist = {}
    phottablelist[ixs_im[0]]=image

    # Merge the tables into one mastertable
    matches = pdastrostatsclass()
    matches.t = pd.concat(phottablelist,ignore_index=True)

    imtable.write()

    converged = False
    counter = 0

    ixs_use = matches.getindices()

    # the excluded indices are saved here
    ixs_excluded =  AnotB(matches.getindices(),ixs_use)

    # start fresh
    ixs4fit= deepcopy(ixs_use)
    poly_degree = order

    print('Fitting wcs polynomial')
    while not converged:
        
        # -- fit only the stars the sigma clipping has kept so far (it used to fit all of them every iteration) -- #
        fit = matches.t.loc[ixs4fit]
        coeff_Sci2IdlX = polyfit0(fit['u'], fit['xprime'], fit['yprime'],order=poly_degree, rotation_fit_x=True,fit_coeffs0=True)
        coeff_Sci2IdlY = polyfit0(fit['v'], fit['xprime'], fit['yprime'],order=poly_degree, rotation_fit_y=True,fit_coeffs0=True)

        #sys.exit(0)

        matches.t['u_fit'] = pysiaf.utils.polynomial.poly(coeff_Sci2IdlX, matches.t['xprime'], matches.t['yprime'], order=poly_degree)
        matches.t['v_fit'] = pysiaf.utils.polynomial.poly(coeff_Sci2IdlY, matches.t['xprime'], matches.t['yprime'], order=poly_degree)
        matches.t['du_pix']  = (matches.t['u'] - matches.t['u_fit']) 
        matches.t['dv_pix']  = (matches.t['v'] - matches.t['v_fit'])

        matches.t['du_as'] = matches.t['du_pix'] * 21.0
        matches.t['dv_as'] = matches.t['dv_pix'] * 21.0

        matches.calcaverage_sigmacutloop('du_pix',indices=ixs4fit,verbose=0,Nsigma=3,percentile_cut_firstiteration=80)
        #print(matches.statparams)
        Nclip = matches.statparams['Nclip']
        ixs4fit = matches.statparams['ix_good']

        matches.calcaverage_sigmacutloop('dv_pix',indices=ixs4fit,verbose=0,Nsigma=3,percentile_cut_firstiteration=80)
        #print(matches.statparams)
        # add the Nclip from y to the Nclip from x
        Nclip += matches.statparams['Nclip']
        ixs4fit = matches.statparams['ix_good']

        counter+=1
        if Nclip<1:
            print('CONVERGED!!!!!!')
            converged = True
        else:
            print(f'iteraton {counter}: {Nclip} clipped, {len(ixs4fit)} kept')
            
        if counter>20:
            converged = True

    ixs_cut_3sigma = AnotB(ixs_use,ixs4fit)    
    print(f'3-sigma clip result: {len(ixs_cut_3sigma)} ({len(ixs_cut_3sigma)/len(ixs_use)*100.0:.1f}%) out of {len(ixs_use)} clipped')
    matches.t['cutflag']=8   ### All 8 should be overwritten by the following commands! If not it's a bug!!
    matches.t.loc[ixs4fit,'cutflag']=0
    matches.t.loc[ixs_cut_3sigma,'cutflag']=1
    matches.t.loc[ixs_excluded,'cutflag']=2


    # limits for plots.
    residual_limits = (-0.4,0.4)

    imIDs = [-1] # -1 is used for all imIDs
    imIDs.extend(unique(matches.t['imID']))
    imIDs.sort()
    for imID in imIDs:
        if imID == -1:
            plot_residuals(matches, ixs4fit, ixs_cut_3sigma, ixs_excluded,
                        residual_limits = residual_limits)
            plt.savefig(f'{save_folder}/residuals_fullfit.png')
        else:
            continue

    
    (coeff_Idl2SciX,coeff_Idl2SciY) = fit_Idl2Sci(NAXIS1,NAXIS2,CRPIX1,CRPIX2,coeff_Sci2IdlX,coeff_Sci2IdlY,order)
    
    coeffs = make_coeff_table(coeff_Sci2IdlX,coeff_Sci2IdlY,coeff_Idl2SciX,coeff_Idl2SciY,order)

    coeffs.write(f'{save_folder}/polyfit_coeffs.txt')

    plt.close('all')

    del matches
    del image
    del ccd_sources
    del header
    del coeff_Sci2IdlX
    del coeff_Sci2IdlY

    gc.collect()

def Update_SIP_Header(save_folder,update_offset=True,update_rotation=True,update_distortion=True):
    """
    Update the SIP distortion coefficients in a FITS file header using coefficients from a file.
    
    Parameters
    ----------
    fitsfilename : str
        Path to the input FITS file
    coeff_filename : str
        Path to the coefficient file (output from make_coeff_table)
    output_fitsfilename : str, optional
        Path to save the updated FITS file. If None, overwrites the input file.
    """

    coeff_table = pd.read_csv(f'{save_folder}/polyfit_coeffs.txt', sep=r"\s+")

    linear = coeff_table[coeff_table['exponent_x'] + coeff_table['exponent_y'] < 2].copy()
    
    coeff_table = coeff_table[coeff_table['exponent_x'] + coeff_table['exponent_y'] >= 2].copy()

    max_order = int((coeff_table['exponent_x'] + coeff_table['exponent_y']).max())


    fitsfilename = f'{save_folder}/corrected.fits'

    with fits.open(fitsfilename) as hdul:
        header = hdul[1].header

        if update_offset:
            # offset terms (exponent_x=0, exponent_y=0)
            offset = linear[linear['exponent_x'] + linear['exponent_y'] == 0].iloc[0]
            # adjust CRPIX by the offset
            header['CRPIX1'] += offset['Sci2IdlX']
            header['CRPIX2'] += offset['Sci2IdlY']

        if update_rotation:
            # rotation/scale terms (exponent_x+exponent_y == 1)
            rot = linear[linear['exponent_x'] + linear['exponent_y'] == 1]
            # row with exponent_x=1, exponent_y=0 gives x terms
            xx = rot[rot['exponent_x'] == 1].iloc[0]
            # row with exponent_x=0, exponent_y=1 gives y terms  
            xy = rot[rot['exponent_x'] == 0].iloc[0]

            # update CD matrix by multiplying with the fitted rotation/scale
            cd = np.array([[header['CD1_1'], header['CD1_2']],
                           [header['CD2_1'], header['CD2_2']]])
            
            rot_matrix = np.array([[xx['Sci2IdlX'], xy['Sci2IdlX']],
                                    [xx['Sci2IdlY'], xy['Sci2IdlY']]])
            
            cd_new = cd @ rot_matrix
            
            header['CD1_1'] = cd_new[0, 0]
            header['CD1_2'] = cd_new[0, 1]
            header['CD2_1'] = cd_new[1, 0]
            header['CD2_2'] = cd_new[1, 1]

        if update_distortion:
            for key in list(header.keys()):
                if (key.startswith('A_') or key.startswith('B_') or
                    key.startswith('AP_') or key.startswith('BP_')):
                    if key not in ('A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER'):
                        del header[key]

            header['A_ORDER'] = max_order
            header['B_ORDER'] = max_order
            header['AP_ORDER'] = max_order
            header['BP_ORDER'] = max_order

            for _, row in coeff_table.iterrows():
                ex = int(row['exponent_x'])
                ey = int(row['exponent_y'])

                a_key  = f'A_{ex}_{ey}'
                b_key  = f'B_{ex}_{ey}'
                ap_key = f'AP_{ex}_{ey}'
                bp_key = f'BP_{ex}_{ey}'

                header[a_key]  = (row['Sci2IdlX'], 'distortion coefficient')
                header[b_key]  = (row['Sci2IdlY'], 'distortion coefficient')
                header[ap_key] = (row['Idl2SciX'], 'inv distortion coefficient')
                header[bp_key] = (row['Idl2SciY'], 'inv distortion coefficient')

        hdul.writeto(fitsfilename, overwrite=True)
        hdul.close()

def Final_Shift_Update(save_folder):

    ccd_sources = pd.read_csv(f'{save_folder}/ccd_sourcefits.csv')
    image_path = f'{save_folder}/corrected.fits'
    with fits.open(image_path) as hdul:
        header = hdul[1].header
        wcs = WCS(header)

        x,y = wcs.all_world2pix(ccd_sources.ra,ccd_sources.dec,0)

        offsetx = np.nanmedian(ccd_sources.xPSF-x)
        offsety = np.nanmedian(ccd_sources.yPSF-y)

        header['CRPIX1'] += offsetx
        header['CRPIX2'] += offsety

        hdul.writeto(image_path, overwrite=True)
        hdul.close()
