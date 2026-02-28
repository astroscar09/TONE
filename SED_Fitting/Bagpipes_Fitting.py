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

def fit_instruction_nebular_fixedz(z, model = 'delayed_tau'):
    
    '''
    Function that creates the bagpipes fit model, this is what bagpipes will attempt to fit with.
    
    Input:
    z (float): redshift of the source, but feel free to change this if you do not have redshift
    model (str): 
    returns:
    fit_instruction (dictionary): the intructions BP will use to fit the galaxy
    '''
    
    if model == 'delayed_tau':
        print("Making Fit Instructions for Delayed-Tau SFH Model")
        
        #Model Building 
        model = {}
        model['age'] = (.01, 13)                  # Age of the galaxy in Gyr

        model['tau'] = (.02, 14)                  # Delayed decayed time
        model["metallicity"] = (0., 2)          # Metallicity in terms of Z_sol
        model["massformed"] = (4., 13.)           # log10 of mass formed
        
        dust = {}                                 # Dust component
        dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
        dust["Av"] = (0., 3.)                     # Vary Av between 0 and 3 magnitudes

        #############
        #NOTE: Before next run need to talk to steve about fixing this at 1.44 or letting it be a param
        #      BP will fit
        #############
        dust["eta"] = 1

        #will need to include this, this includes SF regions and their emission to the spectrum
        nebular = {}

        #changed upper limits from -4 to -2 as it seems I keep getting an error with -1
        nebular["logU"] = (-4, -1)

        fit_instructions = {}
        fit_instructions['delayed'] = model
        
        fit_instructions['redshift'] = z
        
        fit_instructions['dust'] = dust
        fit_instructions['nebular'] = nebular
    
    
        return fit_instructions
    
    elif model == 'nonparam':
        #values taken from the PLANCK CMB 2018 paper
        Om0 = .315
        Ode0 = 1 - Om0
        cosmo = LambdaCDM(H0 = 67.4, 
                          Om0 = .315, 
                          Ode0 = Ode0)
        
        age_Gyr = cosmo.age(z).value
        age_Myr = age_Gyr * 1e3
        
        starting_bin = np.array([0])
        bin_end = np.log10(age_Myr) - .05
        
        bins = np.logspace(np.log10(5), bin_end, 9)
        
        age_bins = np.append(starting_bin, bins)
        
        
        print("Making Fit Instructions for Non-Parametric SFH Model")
        dust = {}
        dust["type"] = "Calzetti"
        dust["eta"] = 1.
        dust["Av"] = (0., 3.)

        nebular = {}
        if bp_sf_model == 'bpass':
            nebular["logU"] = (-4, -1)
        else:
            nebular["logU"] = (-4, 0)  

        fit_instructions = {}
        fit_instructions["dust"] = dust
        fit_instructions["nebular"] = nebular
        fit_instructions["redshift"] = z

        #print(age_bins)
        
        continuity = {}
        continuity["massformed"] = (5, 13.)
        continuity["metallicity"] = (0.01, 3.)
        continuity["metallicity_prior"] = "log_10"
        continuity["bin_edges"] = list(age_bins)

        for i in range(1, len(continuity["bin_edges"])-1):
            continuity["dsfr" + str(i)] = (-10., 10.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"

        fit_instructions["continuity"] = continuity
        
        return fit_instructions
        
    elif model == "bursty":
        
        #values taken from the PLANCK CMB 2018 paper
        Om0 = .315
        Ode0 = 1 - Om0
        cosmo = LambdaCDM(H0 = 67.4, 
                          Om0 = .315, 
                          Ode0 = Ode0)
        
        age_Gyr = cosmo.age(z).value
        age_Myr = age_Gyr * 1e3
        
        starting_bin = np.array([0, 5, 10, 50])
        bin_end = np.log10(age_Myr) - .01
        
        bins = np.logspace(np.log10(100), bin_end, 5)
        
        age_bins = np.append(starting_bin, bins)
       
        print("Making Fit Instructions for Bursty Non-Parametric SFH Model \n", flush=True)
        dust = {}
        dust["type"] = "Salim"
        dust["eta"] = 1.
        dust["Av"] = (0.001, 3.)
        dust['Av_prior'] = 'log_10'
        dust["delta"] = (-1.2, 0.4) #taken from Salim et al 2018 Figure 6
        dust["B"] = (0., 4.5) # Taken from Salim et al 2018 as seen in their Figure 3 colorbar

        nebular = {}
        if bp_sf_model == 'bpass':
            nebular["logU"] = (-4, -1)
        else:
            nebular["logU"] = (-4, 0) 

        fit_instructions = {}
        fit_instructions["dust"] = dust
        fit_instructions["nebular"] = nebular
        fit_instructions["redshift"] = z


        continuity = {}
        continuity["massformed"] = (4, 13.)
        continuity["metallicity"] = (0.01, 3.)
        continuity["metallicity_prior"] = "log_10"
        continuity["bin_edges"] = list(age_bins)

        for i in range(1, len(continuity["bin_edges"])-1):
            continuity["dsfr" + str(i)] = (-10., 10.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"
            
            #adding this prior scale to make it bursty
            continuity["dsfr" + str(i) + "_prior_scale"] =2.0
            
        fit_instructions["continuity"] = continuity
        
        return fit_instructions

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

