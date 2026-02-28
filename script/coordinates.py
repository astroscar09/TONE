from astropy.coordinates import SkyCoord
from astropy import units as u

def make_skycoord(ra, dec):
    '''
    Function to make a skycoord object based off of input ra and dec
    '''

    data_skycoord = SkyCoord(ra = ra, 
                             dec = dec, 
                             unit = 'degree')
    
    return data_skycoord


def search_around_coordinates(phot_skycoord, hdr_skycoord, search_radius):
    '''
    Function to search around the photometric sources to find HDR detections within the input search radius (in arcsec)
    
    Inputs
    ---------
    phot_skycoord: the skycoord object relating to the photometric catalog
    hdr_skycoord:  the skycoord object relating to the HDR catalog
    search_radius: the radius to search around each photometric source for an HDR detection (units of arcsec)
    '''
    idx_hdr, idx_phot, separation, _ = phot_skycoord.search_around_sky(hdr_skycoord, 
                                                                        search_radius*u.arcsecond)
    
    return idx_phot, idx_hdr, separation



def match_catalogs(cat1, df1, idx_cat1, idx_df1, sep, sep_thresh=1.51):

    '''
    Function to match the catalogs based off of the indexes and separation
    Inputs  
    ----------
    cat1: the first catalog to match
    df1: the first dataframe to match
    idx_cat1: the indexes of the first catalog
    idx_df1: the indexes of the first dataframe
    sep: the separation between the two catalogs
    '''

    matched_cat = cat1[idx_cat1]
    matched_df = df1.iloc[idx_df1]

    sep_mask = (sep.arcsec < sep_thresh) 

    matched_cat = matched_cat[sep_mask]
    matched_df = matched_df[sep_mask]
    matched_df['separation'] = sep[sep_mask].arcsec
    matched_cat['New_IDs'] = np.arange(1, len(matched_cat) + 1)

    return matched_cat, matched_df

def cross_match(phot_cat, hdr_df, phot_config, selection_config, hdr_ra = 'ra', hdr_dec = 'dec'):

    phot_ra = phot_config['ra_col']
    phot_dec = phot_config['dec_col']
    search_radius = selection_config['cross_match']['search_radius']
    
    hdr_coords  = make_skycoord(hdr_df[hdr_ra], hdr_df[hdr_dec])
    phot_coords = make_skycoord(phot_cat[phot_ra], phot_cat[phot_dec])

    idx_phot, idx_hdr, separation = search_around_coordinates(phot_coords, hdr_coords, search_radius)

    matched_phot, matched_hdr_df = match_catalogs(phot_cat, hdr_df, idx_phot, idx_hdr, separation)

    return matched_phot, matched_hdr_df
