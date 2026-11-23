import os
import pandas as pd
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astroquery.vizier import VizierClass
import numpy as np
from astropy.io import fits 
from astropy.wcs import WCS 
from time import time as t

no_targets_found_message = ValueError('Either no sources were found in the query region '
                                        'or Vizier is unavailable')

# -- VizieR mirrors for every catalogue query in tessellate, tried in order: a mirror that is down, or that reports a
# -- catalogue unavailable, falls through to the next. Strasbourg was down in Nov 2026, so the CfA mirror goes first.
# -- Override with a comma list in the TESSELLATE_VIZIER environment variable.
VIZIER_SERVERS = [s.strip() for s in os.environ.get('TESSELLATE_VIZIER',
                                                    'vizier.cfa.harvard.edu,vizier.cds.unistra.fr').split(',')]
VIZIER_TIMEOUT = 60         # seconds per mirror and query
VIZIER_RETRIES = 5          # attempts per mirror while it answers "database not currently reachable" (the CfA
VIZIER_RETRY_DELAY = 5      # mirror flips in and out of that, Nov 2026), VIZIER_RETRY_DELAY seconds apart
# -- If no mirror can serve a catalogue, raise (default) rather than quietly treat it as empty. Set True (or
# -- TESSELLATE_VIZIER_ALLOW_MISSING=1) to warn and carry on without it, e.g. building variable catalogues
# -- while one source catalogue (ASAS-SN on the CfA mirror, Nov 2026) is unreachable.
VIZIER_ALLOW_MISSING = os.environ.get('TESSELLATE_VIZIER_ALLOW_MISSING', '0') == '1'


class _HttpsVizier(VizierClass):
    """astroquery's Vizier over https: older astroquery (0.4.6) always builds an http:// URL, and the mirrors
    redirect that to https, which drops the POSTed query -- the result comes back silently empty."""

    def _server_to_url(self, *args, **kwargs):
        return super()._server_to_url(*args, **kwargs).replace('http://', 'https://', 1)


def _vizier_unavailable(server, catalog, coord, radius):
    """True if this mirror says it can't serve the catalogue (an empty result is then not a real 'no sources')."""
    import requests
    try:
        r = requests.get(f'https://{server}/viz-bin/votable', timeout=VIZIER_TIMEOUT,
                         params={'-source': catalog, '-c': f'{coord.ra.deg} {coord.dec.deg}',
                                 '-c.rs': f'{Angle(radius).arcsec:.1f}', '-out.max': '1'})
    except Exception:
        return True
    text = r.text
    return ('not currently reachable' in text or 'no connection' in text or
            ('QUERY_STATUS" value="ERROR' in text and 'OVERFLOW' not in text))


def query_vizier(coords, catalog, radius, **kwargs):
    """
    Vizier.query_region(coords, catalog=catalog, radius=radius) on the first mirror in VIZIER_SERVERS that answers
    and can serve the catalogue; kwargs (columns, column_filters, ...) go to the Vizier object. Returns astroquery's
    TableList, empty if the region has no sources. Raises RuntimeError if no mirror can serve the catalogue
    (or warns and returns an empty TableList with VIZIER_ALLOW_MISSING).
    """
    from time import sleep
    from astroquery.utils import TableList
    coord = coords if coords.isscalar else coords[0]
    problems = []
    for server in VIZIER_SERVERS:
        for attempt in range(VIZIER_RETRIES):
            try:
                v = _HttpsVizier(vizier_server=server, timeout=VIZIER_TIMEOUT, row_limit=-1, **kwargs)
                result = v.query_region(coords, catalog=catalog, radius=radius, cache=False)
            except Exception as e:          # mirror down or not answering: no point retrying it
                problems.append(f'{server}: {type(e).__name__}')
                break
            if result is not None and len(result) > 0:
                return result
            if not _vizier_unavailable(server, catalog, coord, radius):
                return TableList([]) if result is None else result     # answered: genuinely no sources
            sleep(VIZIER_RETRY_DELAY)
        else:
            problems.append(f'{server}: catalogue unavailable after {VIZIER_RETRIES} tries')
    message = f'VizieR catalogue {catalog} could not be queried ({"; ".join(problems)})'
    if VIZIER_ALLOW_MISSING:
        print(f'WARNING: {message} -- carrying on without it')
        return TableList([])
    raise RuntimeError(message)


def Get_Catalogue(ra,dec,size,Catalog = 'gaia'):
	"""
	Get the coordinates and mag of all sources in the field of view from a specified catalogue.

	I/347/gaia2dis   Distances to 1.33 billion stars in Gaia DR2 (Bailer-Jones+, 2018)

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		Catalogue 			str 	Permitted options: 'gaia', 'dist', 'ps1'
	
	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	c1 = SkyCoord(ra, dec, frame='icrs', unit='deg')
	# Use pixel scale for query size
	pix_scale = 21.0
	# We are querying with a diameter as the radius, overfilling by 2x.
	if Catalog == 'gaia':
		catalog = "I/345/gaia2"
	elif Catalog == 'dist':
		catalog = "I/350/gaiaedr3"
	elif Catalog == 'ps1':
		catalog = "II/349/ps1"
	elif Catalog == 'skymapper':
		catalog = 'II/358/smss'
	else:
		raise ValueError(f"{catalog} not recognised as a catalog. Available options: 'gaia', 'dist','ps1'")
	if Catalog == 'gaia':
		result = query_vizier(c1, catalog, Angle(size * pix_scale + 60, "arcsec"), column_filters={'Gmag':'<19'})
	else:
		result = query_vizier(c1, catalog, Angle(size * pix_scale + 60, "arcsec"))

	no_targets_found_message = ValueError('Either no sources were found in the query region '
										  'or Vizier is unavailable')
	#too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
	if result is None:
		raise no_targets_found_message
	elif len(result) == 0:
		raise no_targets_found_message
	result = result[catalog].to_pandas()
	
	return result 


def Get_Gaia(ra,dec,size,wcsObj,magnitude_limit = 18, Offset = 10,verbose=False):
    """
    Get the coordinates and mag of all gaia sources in the field of view.

    -------
    Inputs-
    -------
        tpf 				class 	target pixel file lightkurve class
        magnitude_limit 	float 	cutoff for Gaia sources
        Offset 				int 	offset for the boundary 

    --------
    Outputs-
    --------
        coords 	array	coordinates of sources
        Gmag 	array 	Gmags of sources 
    """
    keys = ['objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','gKmag','e_gKmag','rmag',
            'e_rmag','rKmag','e_rKmag','imag','e_imag','iKmag','e_iKmag','zmag','e_zmag','zKmag','e_zKmag',
            'ymag','e_ymag','yKmag','e_yKmag','tmag','gaiaid','gaiamag','gaiadist','gaiadist_u','gaiadist_l',
            'row','col']
    
    if verbose:
        ts = t()
        print(f'   getting gaia catalogue in radius {size} px:')
        print('        Gaia...',end='\r')
    result = Get_Catalogue(ra,dec,size,Catalog='gaia')
    if verbose:
        print(f'       Gaia...Done! ({(t()-ts):.2f}s)')

    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    
    ras = result['RA_ICRS'].values
    decs = result['DE_ICRS'].values

    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T

    # ras = 

    try:
        coords = wcsObj.all_world2pix(ras,decs, 0) ## TODO, is origin supposed to be zero or one?
    except:
        good_coords = []
        for i in range(len(ras)):
            r = ras[i]
            d = decs[i]
        # for i,radec in enumerate(radecs):
            try:
                c = wcsObj.all_world2pix(r,d, 0)
                good_coords.append(i)
            except:
                pass
        ras = ras[good_coords]
        decs = decs[good_coords]
        result = result.iloc[good_coords]
        radecs = radecs[good_coords]

        # result = result.iloc[good_coords]
        coords = wcsObj.all_world2pix(ras,decs, 0) ## TODO, is origin supposed to be zero or one?

    coords = np.array(coords).T

    source = result['Source'].values
    Gmag = result['Gmag'].values
    #Jmag = result['Jmag']
    ind = (((coords[:,0] >= -10) & (coords[:,1] >= -10)) & 
            ((coords[:,0] < (size + 10)) & (coords[:,1] < (size + 10))))
    
    coords = coords[ind]
    radecs = radecs[ind]
    Gmag = Gmag[ind]
    source = source[ind]
    Tmag = Gmag - 0.5

    #Jmag = Jmag[ind]
    return radecs, Tmag, source

def create_external_gaia_cat(centre,size,wcs,save_path,maglim,verbose=False):
     
    ra,dec = centre
    gp,gm, source = Get_Gaia(ra,dec,size,wcs,magnitude_limit=maglim,verbose=verbose)
    gaia = pd.DataFrame(np.array([gp[:,0],gp[:,1],gm,source]).T,columns=['ra','dec','mag','Source'])

    gaia.to_csv(f'{save_path}/local_gaia_cat.csv',index=False)


def search_atlas_var(coords, radius):
    import mastcasjobs

    query = """select o.ATO_ID, o.ra, o.dec, o.CLASS
               from fGetNearbyObjEq("""+str(coords[0])+','+str(coords[1])+","+str(radius)+""") as nb
               inner join object as o on o.objid = nb.ObjID
               order by o.objid
            """
    try:
        jobs = mastcasjobs.MastCasJobs(context="HLSP_ATLAS_VAR")
        results = jobs.quick(query, task_name="TESSELLATE Atlas var cat")
        results = results.to_pandas()
    except:
        results = None
    return results



def cross_match(obs_cat, viz_cat,tol=2*21,variable=True):
    dist = np.sqrt((obs_cat.ra[:,np.newaxis] - viz_cat.ra[np.newaxis,:])**2 + (obs_cat.dec[:,np.newaxis] - viz_cat.dec[np.newaxis,:])**2)
    closest = np.argmin(dist,axis=1)
    valid = np.nanmin(dist,axis=1) < tol/60**2
    matched = closest[valid]
    if variable:
        obs_cat['Type'] = 'none'
        obs_cat['Prob'] = 0
        obs_cat['Type'].iloc[valid] = viz_cat['Type'].iloc[matched]
        obs_cat['Prob'].iloc[valid] = viz_cat['Prob'].iloc[matched]
    else:
        obs_cat['GaiaID'] = 0 
        obs_cat['GaiaID'].iloc[valid] = viz_cat['Source'].iloc[matched]
    return obs_cat
 
def get_catalog(catalog, centre, radius,gaia=False):
    coords = SkyCoord(ra=centre[0]*u.deg,
                      dec=centre[1]*u.deg)
    if gaia:
        t_result = query_vizier(coords, catalog, radius*u.deg, column_filters={'Gmag':'<19.5'})
    else:
        t_result = query_vizier(coords, catalog, radius*u.deg)
    if len(t_result) > 0:
        return t_result[catalog].to_pandas()
    else:
        return None
 
# def find_variables(coords,viz_cat,radius):
#     # Fetch and save catalogs from Vizier.
#     # Gaia Variable Catalog
#     varisum = get_catalog("I/358/varisum",coords,radius)
#     t.sleep(10)
#     try:
#         atlas = search_atlas_var(coords,radius)
#     except:
#         atlas = None
#     t.sleep(10)
#     # ASASN Variable Catalog
#     asasn = get_catalog("II/366/catalog",coords,radius)
#     t.sleep(10)
#     # DES RRLyrae Catalog
#     des_var = get_catalog("J/AJ/158/16/table11",coords,radius)
#     if (varisum is None) & (asasn is None) & (des_var is None) & (atlas is None):
#         print("No known variables found ...")
#         variables = pd.DataFrame(columns=['ra','dec','Type','Prob'])
#     else:
#         if varisum is not None:
#             varisum['Type'] = 'None'
#             varisum['Prob'] = 0
#             keys = list(varisum.keys())[12:-2]
#             for key in keys:
#                 t = varisum[key].values > 0
#                 varisum['Type'].iloc[t] = key
#                 varisum['Prob'].iloc[t] = varisum[key].values[t]
#             varisum = varisum.rename(columns={'RA_ICRS': 'ra',
#                                               'DE_ICRS': 'dec'})
#             varisum = pd.DataFrame(varisum, columns=['ra','dec','Type','Prob'])
#             varisum['Catalog'] = 'I/358/varisum'
#         if atlas is not None:
#             atlas = atlas.rename(columns={'CLASS':'Type'})
#             atlas['Prob'] = 1
#             atlas['Catalog'] = 'HLSP_ATLAS_VAR'
#         if asasn is not None:
#             asasn = asasn.rename(columns={'RAJ2000': 'ra',
#                                           'DEJ2000': 'dec'})
#             asasn = pd.DataFrame(asasn, columns=['ra','dec','Type','Prob'])
#             asasn['Catalog'] = 'II/366/catalog'
#         else:
#             asasn = None
 
#         if des_var is not None:
#             des_var = des_var.rename(columns={'RAJ2000': 'ra',
#                                               'DEJ2000': 'dec'})
#             des_var = pd.DataFrame(des_var, columns=['ra','dec'])
#             des_var['Type'] = 'RRLyrae'
#             des_var['Prob'] = 1
#             des_var['Catalog'] = 'DES'
#         else:
#             des_var = None
 
#         variables = pd.concat([varisum, asasn, des_var])
    
#     obs = cross_match(viz_cat, variables)
#     return obs

def gaia_stars(obs_cat,size=2*21,mag_limit=19.5):
    coords = SkyCoord(ra=obs_cat.ra.values*u.deg,
                      dec=obs_cat.dec.values*u.deg)
    gaia = query_vizier(coords, "I/355/gaiadr3", Angle(size, "arcsec"), column_filters={'Gmag':f'<{mag_limit}'})
    gaia = gaia['I/355/gaiadr3'].to_pandas()
    gaia = gaia.rename(columns={'RA_ICRS': 'ra',
                                'DE_ICRS': 'dec'})
    obs = cross_match(obs_cat,gaia,tol=size,variable=False)
    return obs 

####### Download variables

def get_variable_cats(coords,radius,verbose):
    from time import time as t

    if verbose:
        ts = t()
        print(f'   getting variable catalogues in radius {radius:.1f} deg:')
        print('       Varisum...',end='\r')


    varisum = get_catalog("I/358/varisum",coords,radius)
    if verbose:
        print(f'       Varisum...Done! ({(t()-ts):.1f}s)')
        print(f'       Atlas...',end='\r')
        ts = t()

    atlas = search_atlas_var(coords,radius)
    if verbose:
        print(f'       Atlas...Done! ({(t()-ts):.1f}s)')
        print(f'       ASASN...',end='\r')
        ts = t()

    asasn = get_catalog("II/366/catalog",coords,radius)
    if verbose:
        print(f'       ASASN...Done! ({(t()-ts):.1f}s)')
        print(f'       DES...',end='\r')
        ts = t()

    des_var = get_catalog("J/AJ/158/16/table11",coords,radius)
    if verbose:
        print(f'       DES...Done! ({(t()-ts):.1f}s)')
        print('\n')

    if (varisum is None) & (asasn is None) & (des_var is None) & (atlas is None):
        print("No known variables found ...")
        variables = pd.DataFrame(columns=['ra','dec','Type','Prob'])
        return variables
    else:
        if varisum is not None:
            varisum['Type'] = 'None'
            varisum['Prob'] = 0
            keys = list(varisum.keys())[12:-2]
            for key in keys:
                tt = varisum[key].values > 0
                varisum.loc[tt,'Type'] = key
                varisum.loc[tt, 'Prob'] = varisum[key].values[tt]
            varisum = varisum.rename(columns={'RA_ICRS': 'ra',
                                              'DE_ICRS': 'dec',
                                              'Source': 'ID'})
            varisum = pd.DataFrame(varisum, columns=['ra','dec','Type','Prob','ID'])
            varisum['Catalog'] = 'I/358/varisum'
        if atlas is not None:
            atlas = atlas.rename(columns={'CLASS':'Type','ATO_ID':'ID'})
            atlas['Prob'] = 1
            atlas['Catalog'] = 'HLSP_ATLAS_VAR'
        if asasn is not None:
            asasn = asasn.rename(columns={'RAJ2000': 'ra',
                                          'DEJ2000': 'dec',
                                          'Gaia': 'ID'})
            asasn = pd.DataFrame(asasn, columns=['ra','dec','Type','Prob','ID'])
            asasn['Catalog'] = 'II/366/catalog'
            asasn.loc[asasn['Type'] == 'ROT:', 'Type'] = 'ROT'
        else:
            asasn = None
 
        if des_var is not None:
            des_var = des_var.rename(columns={'RAJ2000': 'ra',
                                              'DEJ2000': 'dec'})
            des_var = pd.DataFrame(des_var, columns=['ra','dec','ID'])
            des_var['Type'] = 'RRLyrae'
            des_var['Prob'] = 1
            des_var['Catalog'] = 'DES'
        else:
            des_var = None
        tables = [varisum,asasn,des_var,atlas]
        variables = None
        for tab in tables:
            variables = join_cats(variables,tab)

    return variables


def create_external_var_cat(center,size,save_path,verbose=False):
    varcat = get_variable_cats(center,size,verbose=verbose)
    from .tools import save_table
    save_table(varcat,save_path+'/variable_catalog.csv')


def join_cats(obs_cat, viz_cat,rad = 2):
    if obs_cat is not None:
        if viz_cat is not None:
            radius_threshold = rad*u.arcsec
            coords_viz = SkyCoord(ra=viz_cat.ra, dec=viz_cat.dec, unit='deg')
            coords_obs = SkyCoord(ra=obs_cat.ra, dec=obs_cat.dec, unit='deg')
            idx, d2d, d3d = coords_viz.match_to_catalog_3d(coords_obs)
            sep_constraint = d2d <= radius_threshold
            # Get entries in cat_ref with a match
            viz_matched = viz_cat[~sep_constraint]
            # Get matched entries in cat_sci
            joined = pd.concat([obs_cat,viz_matched],ignore_index=True)
            # re-index to match two dfs
            joined = joined.reset_index(drop=True)
        else:
            joined = obs_cat
    else:
        joined = viz_cat
    return joined


#def cross_match_DB(cat1,cat2,radius):
#    coords1 = SkyCoord(ra=cat1['ra'].values, dec=cat1['dec'].values, unit='deg')
#    coords1 = SkyCoord(ra=cat2['ra'].values, dec=cat2['dec'].values, unit='deg')
#    idx, d2d, d3d = cat1.match_to_catalog_3d(cat2)
#    sep_constraint = d2d <= radius_threshold

#    cat1_id = sep_constraint
#    cat2_id = idx[sep_constraint]
#    return cat1_id,cat2_id

def cross_match_DB(cat1,cat2,distance=2*21,njobs=-1):
    from sklearn.cluster import DBSCAN

    all_ra = np.append(cat1['ra'].values,cat2['ra'].values)
    all_dec = np.append(cat1['dec'].values,cat2['dec'].values)
    cat2_ind = len(cat1)

    p = np.array([all_ra,all_dec]).T
    cluster = DBSCAN(eps=distance/60**2,min_samples=2,n_jobs=njobs).fit(p)
    labels = cluster.labels_
    unique_labels = set(labels)
    cat1_id = []
    cat2_id = []

    for label in unique_labels:
        if label > -1:
            inds = np.where(labels == label)[0]
            if (inds < cat2_ind).any() & (inds > cat2_ind).any():
                if len(inds) > 2:
                    dra = all_ra[np.where(labels == label)[0]]
                    ddec = all_dec[np.where(labels == label)[0]]
                    d = (dra - dra[0])**2 + (ddec - ddec[0])**2
                    args = np.argsort(d)
                    good = inds[args] - cat2_ind > 0
                    good[0] = True
                    args = args[good]
                    inds = inds[args[:2]]
                cat1_id += [inds[0]]
                cat2_id += [inds[1] - cat2_ind]
    return cat1_id,cat2_id

def cross_match_tree(cat1,cat2,distance=2,ax1='ra',ax2='dec'):
    from sklearn.neighbors import KDTree

    p1 = np.array([cat1[ax1].values,cat1[ax2].values]).T.astype(float)
    p2 = np.array([cat2[ax1].values,cat2[ax2].values]).T.astype(float)

    tree = KDTree(p1)
    dist,ind = tree.query(p2, k=1)
    ind = ind.flatten()
    dist = dist.flatten()

    d_ind = np.where(dist < distance)[0]
    min_dist = np.argmin(dist[d_ind])
    d_ind = d_ind[min_dist]
    
    cat2_ind = d_ind
    cat1_ind = ind[cat2_ind]

    return cat1_ind,cat2_ind

def match_result_to_cat(result,cat,columns=['Type','Prob'],distance=2*21,min_entry=2):
    ids = np.unique(result['objid'].values)
    ra = []; dec = []
    Id = []
    for id in ids:
        if len(result.iloc[result['objid'].values == id]) >= min_entry:
            Id += [id]
            ra += [result.loc[result['objid'] == id, 'ra_source'].mean()]
            dec += [result.loc[result['objid'] == id, 'dec_source'].mean()]
    pos = {'objid':Id,'ra':ra,'dec':dec}
    pos = pd.DataFrame(pos)
    id_ind, cat_ind = cross_match_DB(pos,cat,distance) #cross_match_tree(pos,cat,distance/60**2)
    obj_id = pos['objid'].values[id_ind]
    for column in columns:
        result[column] = 0
    if isinstance(obj_id,int) | isinstance(obj_id,np.int64):
        obj_id = [obj_id]
    if isinstance(cat_ind,int) | isinstance(cat_ind,np.int64):
        cat_ind = [cat_ind]
    for i in range(len(obj_id)):
        for column in columns:
            result.loc[result['objid']==obj_id[i],column] = cat[column].values[cat_ind[i]]
    return result 


def serious_source_joining(sf,sd,radius=21*1.5):
    """
    Takes 2 dataframes sf and sd which need to contain the following columns:
    objid, sig, method, frane
    """
    
    avg_sf = weighted_average(sf)
    avg_sf = avg_sf[np.isfinite(avg_sf.ra)]

    avg_sd = weighted_average(sd)
    avg_sd = avg_sd[np.isfinite(avg_sd.ra)]
    # do the matching 
    radius_threshold = radius*u.arcsec
    coords_sf = SkyCoord(ra=avg_sf.ra.values, dec=avg_sf.dec.values, unit='deg')
    coords_sd = SkyCoord(ra=avg_sd.ra.values, dec=avg_sd.dec.values, unit='deg')
    idx, d2d, d3d = coords_sf.match_to_catalog_3d(coords_sd)
    sep_constraint = d2d <= radius_threshold
    # sort out indicies

    matched_ids = zip(avg_sd.loc[idx[sep_constraint],'objid'].values,avg_sf.loc[sep_constraint,'objid'].values)
    unmatched_ids = avg_sd.loc[idx[~sep_constraint],'objid'].values
    replace_dict = dict(matched_ids)
    sd['objid'] = sd['objid'].replace(replace_dict)
    new_ids = np.arange(1,len(unmatched_ids)) + sf['objid'].max()
    replace_dict = dict(zip(unmatched_ids,new_ids))
    sd['objid'] = sd['objid'].replace(replace_dict)
    b = pd.concat([sf,sd])    
    b_cleaned = b[~b.duplicated(subset=['objid', 'frame'], keep='first')]
    b_cleaned
    
    return b_cleaned