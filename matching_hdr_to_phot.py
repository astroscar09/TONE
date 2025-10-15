import sys
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table, unique
from astropy import units as u
from astropy.coordinates import SkyCoord
from hetdex_api.detections import Detections
from hetdex_api.config import HDRconfig
import tables as tb
import matplotlib.pyplot as plt
import os.path as op
import time
from astropy.table import Column, MaskedColumn
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def grab_useful_columns(field_name):
    
    """ Function to grab the useful columns from the config file for a given field. 
    Parameters
    ----------
    field_name : str

        The name of the field for which to grab the useful columns.

    Returns
    -------
    list
        A list of useful columns for the specified field.
    """
    try:
        field_data = config[field_name]
        useful_columns = field_data['useful_columns']
        return useful_columns
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")

def grab_flux_columns(field_name):
    """ Function to grab the flux columns from the config file for a given field.
    Parameters
    ----------
    field_name : str
        The name of the field for which to grab the flux columns.
    Returns
    -------
    tuple
        A tuple containing the flux columns and their corresponding error columns for the specified field.
    """
    try:
        field_data = config[field_name]
        flux_cols = field_data['flux_columns']
        fluxerr_cols = field_data['fluxerr_columns']
        return flux_cols, fluxerr_cols
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")

def get_field_info(field_name):
    
    try:
        
        field_data = config[field_name]
        filters = field_data['filters']
        phot_file = field_data['photometry']
        flux_cols = field_data['flux_columns']
        fluxerr_cols = field_data['fluxerr_columns']
        useful_columns = field_data['useful_columns']
        
        return filters, phot_file, flux_cols, fluxerr_cols, useful_columns
    
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")


#current fields that this will work on is NEP, SHELA, CEERS, PRIMER-UDS, and PRIMER-COSMOS.

def generate_lyaz(hdr_df, ids):
    
    lya_rest = 1215.67
    obs_wave = hdr_df.loc[ids, 'wave'].values
    wave_err = hdr_df.loc[ids, 'wave_err'].values
    
    ratio = obs_wave/lya_rest
    ratio_err = np.abs(ratio)*(wave_err/obs_wave)
    
    lyaz = (ratio) - 1
    
    hdr_df['lya_z'] = lyaz
    hdr_df['lya_z_err'] = ratio_err
    
    return hdr_df

def read_HDR_data(field, hdr_version = '5.0.1'):
    
    '''
    Function to read in the HDR information for the TESLA Survey for a given field.
    
    Input
    -----------------
    field(str): the name of the field we want to find HDR detections in
    hdr_version(string, optional): The HDR version to select the detetection database default to 4.0.0
    
    
    '''
    
    if field.lower() == 'nep':
        
        field = 'nep'
        
    elif field == 'NEP_EUCLID':
        field = 'nep'

    elif field == 'TONE':
        field = 'nep'
        
    elif field.lower() == 'euclid':
        
        field = 'nep'
        
    elif field.lower() == 'shela':
        
        field = 'dex-fall'
        
    elif field.lower() == 'ceers':
        
        field = 'dex-spring'
        
    elif field.lower() == 'primer-cosmos':
        
        field = 'cosmos'

    elif field.lower() == 'cosmos':
        
        field = 'cosmos'
        
    elif field.lower() == 'goodsn':
        
        field = 'goods-n'
        
    elif field.lower() == 'primer-uds':
        
        field = 'other'

    elif field.lower() == 'uds':
        
        field = 'other'
        
    print(f'Reading in HDR data for field: {field}')
    

    hdrv = 'hdr{}'.format(hdr_version[0])
    config = HDRconfig()
    catfile = op.join(config.hdr_dir[hdrv], 'catalogs', 'source_catalog_' + hdr_version + '.fits')
    source_table = Table.read(catfile)

    sel_best = (source_table['flag_best']==1) * (source_table['flag_seldet'] == 1) * (source_table['flag_shot_cosmology']==1)
    sel_apcor = (source_table['apcor'] > 0.4)

    sel_cut1 = (source_table['sn_rres'] >= 4.2)
    sel_cut2 = np.invert((source_table['p_conf'] < 0.5) * (source_table['sn'] >= 5.5))
    sel_cut3 = source_table['CNN_Score_2D_Spectra'] >= 0.2
    sel_cuts = sel_cut1 * sel_cut2 * sel_cut3

    sel_good = (source_table['flag_best'] == 1) #& (source_table['flag_erin_cuts']==1)

    sel_det = source_table['selected_det'] == True
    #We remove any AGN sources
    sel_agn = source_table['source_type']!= 'agn'
    #we also remove continuum sources
    sel_cont = source_table['det_type'] != 'cont'
    #we subselect only detections within the field user input
    field_flag = source_table['field'] == field


    #we make a master mask
    sel_cat = (sel_best & sel_apcor &  sel_cuts & sel_cont & sel_agn) & sel_det & sel_good & field_flag

    #apply mask and remove duplicates based off of source_id
    uniq_table = unique(source_table[sel_cat], keys='source_id')

    #turn that into a DF
    HETDEX_DF = uniq_table.to_pandas()

    HETDEX_DF = generate_lyaz(HETDEX_DF, HETDEX_DF.index.values)

    HETDEX_DF['dex_flags'] = [f'0x{flag:08x}' for flag in HETDEX_DF['flags'].values]
    
    #p#rint(f'HDR Data Loaded Successfully after {tot_time:.2f} seconds!')

    return HETDEX_DF
    
    
def read_in_photom_cat(filename):
    
    '''
    Function that reads in the photometric catalog
    '''
    start = time.time()
    print('Reading in Photometry Data!')
    phot_cat_tab = Table.read(filename)
    end = time.time()
    print(f'Photometry Data Loaded Successfully after {end-start:.2f} seconds!')
    
    return phot_cat_tab


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

def main(photom_file, field, ra_col, dec_col, units = 'muJy', search_radius=1.51, hdr_version='5.0.1'):
    """
    Main function to read in the photometric catalog, HDR data, and match them.
    Parameters:
        photom_file (str): Path to the photometric catalog file.
        field (str): Name of the field to match.
        search_radius (float): Search radius in arcseconds for matching.
        hdr_version (str): Version of the HDR data to use.
    """
    # Read in the photometric catalog
    phot_cat = read_in_photom_cat(photom_file)
    # Read in the HDR data
    hdr_df = read_HDR_data(field, hdr_version)
    # Create SkyCoord objects for both catalogs
    phot_skycoord = make_skycoord(phot_cat[ra_col].data, phot_cat[dec_col].data)
    hdr_skycoord = make_skycoord(hdr_df['ra'].values, hdr_df['dec'].values)
    # Search for matches
    idx_phot, idx_hdr, separation = search_around_coordinates(phot_skycoord, hdr_skycoord, search_radius)
    # Match the catalogs
    matched_cat, matched_df = match_catalogs(phot_cat, hdr_df, idx_phot, idx_hdr, separation)

    cols = grab_useful_columns(field)
    flux_cols, fluxerr_cols = grab_flux_columns(field) 

    if units == 'nJy':
        for f, ferr in zip(flux_cols, fluxerr_cols):
        
            matched_cat[f] = matched_cat[f].data/1000 #converting to muJy
            matched_cat[ferr] = matched_cat[ferr].data/1000 #converting to muJy
    photom_cat = matched_cat[cols]
    photom_cat['Redshift'] = matched_df['lya_z'].values

    return photom_cat, matched_df


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Match photometric catalog to HDR data.")
    parser.add_argument('--photom_file', type=str, help='Path to the photometric catalog file.')
    parser.add_argument('--field', type=str, help='Name of the field to match.')
    parser.add_argument('--ra_col', type=str, default='ra', help='Column name for right ascension in the photometric catalog.')
    parser.add_argument('--dec_col', type=str, default='dec', help='Column name for declination in the photometric catalog.')
    parser.add_argument('--search_radius', type=float, default=1.51, help='Search radius in arcseconds for matching.')
    parser.add_argument('--hdr_version', type=str, default='5.0.1', help='Version of the HDR data to use.')
    
    args = parser.parse_args()
    photom_file = args.photom_file
    field = args.field
    ra_col = args.ra_col
    dec_col = args.dec_col
    search_radius = args.search_radius
    hdr_version = args.hdr_version
    
    photom_cat, matched_df = main(photom_file, field,
                                  ra_col, dec_col, 
                                  search_radius, hdr_version)

    # Save the matched photometric catalog to a CSV file
    photom_cat.write(f'Matched_Catalogs/{field}_matched_photom_cat.fits')
    Table.from_pandas(matched_df).to_csv(f'Matched_Catalogs/{field}_matched_hdr_df.fits')




# 


