import time 
from astropy.table import Table, unique
import numpy as np
from hetdex_api.config import HDRconfig
import os.path as op
import yaml 

def read_config_yaml(file):
    #'../configuration/config.yaml'
    with open(file, 'r') as f:
        config = yaml.safe_load(f)

    return config

def save_table(tab, path):

    tab.write(path, overwrite = True)

def save_df(df, path):
    df.to_csv(path, sep = ' ', index = False)


def read_HDR5(field, config):
    
    '''
    Function to read in the HDR information for the TESLA Survey for a given field.
    
    Input
    -----------------
    field(str): the name of the field we want to find HDR detections in
    hdr_version(string, optional): The HDR version to select the detetection database default to 4.0.0
    
    
    '''

    mapper = {'nep': 'nep', 'NEP_EUCLID': 'nep', 'tone':'nep', 'TONE': 'nep', 'euclid': 'nep', 
              'shela': 'dex-fall', 'ceers': 'dex-spring', 
              'primer-cosmos': 'cosmos', 'cosmos': 'cosmos', 
              'goodsn': 'goods-n', 
              'primer-uds': 'other', 'uds': 'uds'}
    
    hdr5_config = config['HDR5']
    hdr_version = hdr5_config['hdr_version']

    field = mapper.get(field.lower(), None)

    if field is None:
        raise NameError('Field Provided is not in the field Mapper')
        
    print(f'Reading in HDR data for field: {field}')
    

    hdrv = 'hdr{}'.format(hdr_version[0])
    config = HDRconfig()
    catfile = op.join(config.hdr_dir[hdrv], 'catalogs', 'source_catalog_' + hdr_version + '.fits')
    source_table = Table.read(catfile)

    sel_best = (source_table['flag_best']==hdr5_config['flag_best']) * (source_table['flag_seldet'] == hdr5_config['flag_seldet']) * (source_table['flag_shot_cosmology']==hdr5_config['flag_shot_cosmology'])
    sel_apcor = (source_table['apcor'] > hdr5_config['apcor'])

    sel_cut1 = (source_table['sn_rres'] >= hdr5_config['sn_rres'])
    sel_cut2 = np.invert((source_table['p_conf'] < hdr5_config['p_conf']) * (source_table['sn'] >= hdr5_config['sn']))
    sel_cut3 = source_table['CNN_Score_2D_Spectra'] >= hdr5_config['cnn_score']
    sel_cuts = sel_cut1 * sel_cut2 * sel_cut3

    sel_agn = np.ones(len(sel_apcor), dtype = bool)
    sel_cont = np.ones(len(sel_apcor), dtype = bool)

    if hdr5_config['selected_det']:
        sel_det = source_table['selected_det'] == True
    else:
        sel_det = source_table['selected_det'] == False

    if not hdr5_config['agn_included']:
        sel_agn = source_table['source_type']!= 'agn'

    if not hdr5_config['cont_included']:
        #we remove continuum sources
        sel_cont = source_table['det_type'] != 'cont'
    
    #we subselect only detections within the field user input
    field_flag = source_table['field'] == field

    #we make a master mask
    sel_cat = (sel_best & sel_apcor &  sel_cuts & sel_cont & sel_agn) & sel_det & field_flag

    #apply mask and remove duplicates based off of source_id
    uniq_table = unique(source_table[sel_cat], keys='source_id')

    #turn that into a DF
    HETDEX_DF = uniq_table.to_pandas()

    #HETDEX_DF = generate_lyaz(HETDEX_DF, HETDEX_DF.index.values)
    #HETDEX_DF['dex_flags'] = [f'0x{flag:08x}' for flag in HETDEX_DF['flags'].values] 
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