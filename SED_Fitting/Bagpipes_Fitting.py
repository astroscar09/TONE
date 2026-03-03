import bagpipes as pipes
import pandas as pd
from astropy.io import fits
import numpy as np
import time
from astropy.cosmology import LambdaCDM
from astropy.table import Table
import yaml
import argparse
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import os


print('Grabbing the necessary filter curves', flush=True)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_field_info(field_name):
    try:
        field_data = config[field_name]
        filters = field_data['filters']
        phot_file = field_data['photometry']
        flux_cols = field_data['flux_columns']
        fluxerr_cols = field_data['fluxerr_columns']
        bp_sf_model = field_data['bp_sf_model'][0]
        return filters, phot_file, flux_cols, fluxerr_cols, bp_sf_model
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")

def grab_min_and_max_filt_wavelengths(field_name):

    """
    Function to grab the minimum and maximum filter wavelengths for a given field.
    Parameters
    ----------
    field_name : str
        Name of the field (e.g., CEERS, COSMOS, TONE, SHELA, UDS, or GOODSN).
    Returns
    -------
    tuple
        A tuple containing the minimum and maximum filter wavelengths.
    """
    try:
        field_data = config[field_name]
        filter_min = np.array(field_data['filter_min'])
        filter_max = np.array(field_data['filter_max'])

        return filter_min, filter_max
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")
    

parser = argparse.ArgumentParser(description="Run BAGPIPES for a specific field.")
parser.add_argument('--field', type=str, required=True, help='Field name to use (e.g., CEERS, COSMOS, TONE, SHELA, UDS, or GOODSN)')
parser.add_argument('--run_name', type=str, required=True, help='Name of the run')
parser.add_argument('--id', type=int, required=True, help='Object ID to process')

args = parser.parse_args()
field_name = args.field

print(f'Using field: {field_name} \n', flush=True)

filters, phot_file, flux_cols, fluxerr_cols, bp_sf_model = get_field_info(field_name)

print(f'Using the {bp_sf_model} Templates', flush=True)
print(f"{'Filters':<35} {'Flux Columns':<25} {'Flux Error Columns':<25}", flush=True)
print('-' * 85, flush=True)
for filt , f_col, ferr_col in zip(filters, flux_cols, fluxerr_cols):
    print(f'{filt:<35} {f_col:<25} {ferr_col:<25}', flush=True)

filter_min, filter_max = grab_min_and_max_filt_wavelengths(field_name)

Bagpipes_Phot_tab = Table.read(phot_file)
Bagpipes_Phot_DF = Bagpipes_Phot_tab.to_pandas().set_index('New_IDs')


def get_redshift(tab, ID):
    DF = tab.to_pandas().set_index('New_IDs')
    z_tesla = DF.loc[ID, 'Redshift']

    return z_tesla

def Galaxy_Model_Builder(ID, load_func, filters):
    '''
    This function will make a bagpipes galaxy model for one ID
    
    '''
    #need to take the extra step of converting ID to string for bagpipes galaxy function to work properly
    galaxy_bp = pipes.galaxy(str(ID), load_func, filt_list = filters, spectrum_exists=False)
    
    return galaxy_bp


def load_phot(ID):
    
    lya = 1215.67

    ID = int(ID)
    
    zspec = Bagpipes_Phot_DF.loc[ID, 'Redshift']

    obs_lya = lya * (1 + zspec)

    lya_mask = (filter_min < obs_lya) & ( obs_lya < filter_max)

    #getting the full flux and flux error info
    photom_flux = Bagpipes_Phot_DF.loc[ID, flux_cols].values
    photom_flux_err = Bagpipes_Phot_DF.loc[ID, fluxerr_cols].values  
    
    #getting the snr of sources
    snr = photom_flux/photom_flux_err
    
    #if the snr is below -5 then we know it is bad we make the flux 0 and error really big
    bad_flux_idx = snr < -3

    irac_mask = np.array([('ch1' in x) | ('ch2' in x) for x in flux_cols])

    #adding in a 5 % error to the fluxes
    photom_flux_err[~irac_mask] = np.sqrt(photom_flux_err[~irac_mask]**2 + (0.05 * photom_flux[~irac_mask])**2) # we add a 5% error from the fluxes capping the SNR to be at max 20
    photom_flux_err[irac_mask] = np.sqrt(photom_flux_err[irac_mask]**2 + (0.2 * photom_flux[irac_mask])**2)     # we add a 20% error from the fluxes capping the SNR to be at max 10

    #setting bad flux to a really small value and error to be really big
    photom_flux[bad_flux_idx] = 0
    photom_flux_err[bad_flux_idx] = 1e12

    photom_flux[lya_mask] = 0
    photom_flux_err[lya_mask] = 1e12

    TESLA_phot = np.c_[photom_flux.astype(float), photom_flux_err.astype(float)] #* 1000 #adding this to boost the photometry for BOOST_CEERS run

    return TESLA_phot


def fit_BP(index, filters, load_func, z, run, only_fit = True, model = 'bursty'):

    #print('Making the BP Galaxy Model')
    BP_Galaxy = Galaxy_Model_Builder(index, load_func, filters)
    
    #print('Getting the BP Fit Instructions')
    #print(f'Redshift is: {z: .5f}')
    fit_instructions = fit_instruction_nebular_fixedz(z, model = model)
    
    if only_fit:
        
        #start = time.time()
        fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    
        fit.fit(verbose=True)
        #end = time.time()
        del fit, BP_Galaxy, fit_instructions
        import gc	
        gc.collect()
        #duration = end - start
        #print(f'Full Time of the Fit is: {duration:.2f} seconds, {duration/60:.2f} Minutes')
        
    else:
        
        fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    
        fit.fit(verbose=True)
    
        return fit

    
def fit_serial_bp(DF, IDs, run,
                  load_func = load_phot, 
                  filters = filters,
                  only_fit = True, 
                  test = False, model = 'nonparam'):
    
    if test:
        print('Testing the Code on the First 10 Sources')
        
        for idx in IDs[:10]:
            print(f'Fitting Galaxy ID: {idx}')
            z_tesla = get_redshift(DF, idx)
            fit_BP(idx, filters, load_func, only_fit = only_fit, model = model, z = z_tesla, run = run)
        
    else:
        
        print(f'Running on the Full Sample of: {DF.shape[0]} Sources')
        
        for idx in IDs:
            
            z_tesla = get_redshift(DF, idx)
            
            fit_BP(idx, filters, load_func, only_fit = only_fit, z = z_tesla)


def test_functions():
    '''
    Function to test the functions in this script
    '''
    print('Testing the Functions')
    test_ID = 12  # Example ID, replace with a valid one from your dataset
    test_filters = filters
    test_load_func = load_phot
    test_z = get_redshift(Bagpipes_Phot_tab, test_ID)
    print(f'Testing with ID: {test_ID}, Redshift: {test_z:.2f}')
    
    print('Testing Functions', flush=True)
    fit_BP(test_ID, test_filters, test_load_func, test_z, run='test_run', only_fit=True, model='bursty')

    

if __name__ == '__main__':

    #if os.path.exist('pipes/posterior/{}')
    
    print('\nGrabbing Bagpipes Run Name', flush=True)
    run = args.run_name
    print(f'Run-Name: {run} Acquired \n', flush=True)

    
    
    index = args.id
    IDs = Bagpipes_Phot_tab['New_IDs'].data

    ID = IDs[index]
    
    if os.path.exists('pipes/posterior/{run}/{ID}.h5'):
        print(f'Posterior for ID: {ID} already exists, skipping fit.', flush=True)
        exit(0)
    
    else:
        print(f'Posterior for ID: {ID} does not exist, proceeding with fit.', flush=True)
        z_tesla = get_redshift(Bagpipes_Phot_tab, ID)
        
        print(f'About to fit index {index}, ID: {ID} with Redshift: {z_tesla:.2f} \n', flush=True)
        
        fit_BP(ID, filters, load_phot, z_tesla, run, only_fit = True, model = 'bursty')
        
        print(f'Successfully fitted index {index}, ID: {ID} \n', flush=True)

