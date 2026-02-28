
import numpy as np
import pandas as pd
import bagpipes as pipes
from BP_PreLoad import *
from astropy.cosmology import LambdaCDM
from astropy.table import Table
from SED_Fitting.plotting_utils import *
from tqdm import tqdm
from astropy.table import vstack
import os 
import glob as glob
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
except:
    rank = 0

#rank = 0

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

parser = argparse.ArgumentParser(description='Run Bagpipes fitting for a specific field.')
parser.add_argument('--field', type=str, default='TONE', help='Field name to run Bagpipes fitting on.')
parser.add_argument('--id', type = int)
parser.add_argument('--test', action='store_true', help='Run in test mode on first 10 sources.')
parser.add_argument('--merge_summary', action='store_true', help='Merge summary tables from all ranks.')
parser.add_argument('--merge_posteriors', action='store_true', help='Merge posterior tables from all ranks.')
parser.add_argument('--test_merge', action='store_true', help='Test merging of posterior tables.')
args = parser.parse_args()
field_name = args.field.upper()


if rank == 0:
    print(f'Using field: {field_name} \n', flush=True)

filters, phot_file, flux_cols, fluxerr_cols, bp_sf_model = get_field_info(field_name)

if rank == 0:
    print(f'Using the {bp_sf_model} Templates', flush=True)
    print(f"{'Filters':<35} {'Flux Columns':<25} {'Flux Error Columns':<25}", flush=True)
    print('-' * 85, flush=True)
    for filt , f_col, ferr_col in zip(filters, flux_cols, fluxerr_cols):
        print(f'{filt:<35} {f_col:<25} {ferr_col:<25}', flush=True)

filter_min, filter_max = grab_min_and_max_filt_wavelengths(field_name)
Bagpipes_Phot_tab = Table.read(phot_file)


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

    Bagpipes_Phot_DF = Bagpipes_Phot_tab.to_pandas().set_index('New_IDs')
    
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

    #setting bad flux to a really small value and error to be really big
    photom_flux[bad_flux_idx] = 0
    photom_flux_err[bad_flux_idx] = 1e12

    photom_flux[lya_mask] = 0
    photom_flux_err[lya_mask] = 1e12

    TESLA_phot = np.c_[photom_flux.astype(float), photom_flux_err.astype(float)] #* 1000

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
        if rank == 0:
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
    if rank == 0:
        print('Testing Functions', flush=True)
    fit_BP(test_ID, test_filters, test_load_func, test_z, run='test_run', only_fit=True, model='bursty')

    process_one_fit(test_ID)


#####
#this script will take the bagpipes output and perform some posterior analysis
#####

def grab_bagpipes_output(ID, run = field_name):

    model = 'bursty'

    BP_Galaxy = Galaxy_Model_Builder(ID, load_phot, filters)
    z = get_redshift(Bagpipes_Phot_tab, ID)
    fit_instructions = fit_instruction_nebular_fixedz(z, model)
    fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    fit.fit(verbose=True)

    fit.posterior.get_advanced_quantities()

    return fit

def grab_1d_data(fit):
    
    samples_dict = fit.posterior.samples

    posterior_1d = {}
    for key, val in samples_dict.items():
        
        if len(val.shape) == 1:
            #l16, med, u84 = np.percentiles = np.percentile(val, [16, 50, 84])

            posterior_1d[f'{key}'] = val

    tab = Table(posterior_1d)

    return tab

def subselect_plot_tab(tab):
    """Subselects the columns of the table to keep only those relevant for plotting.
    Args:
        tab (astropy.table.Table): The table containing the posterior samples.  
    Returns:
        astropy.table.Table: A new table with only the relevant columns for plotting.
    """
    
    columns_to_keep = ['continuity:metallicity', 'dust:Av', 'dust:B', 'dust:delta', 'nebular:logU', 'stellar_mass', 
                       'sfr', 'ssfr', 'mass_weighted_age', 'mass_weighted_zmet', 'chisq_phot', 'beta', 'Muv', 'burstiness']

    return tab[columns_to_keep]


def grab_model_line_flux(fit, line):

    lower_line = line.lower()
    map_line = {'lya': 'H  1  1215.67A', 
                'halpha': 'H  1  6562.81A', 
                'ha': 'H  1  6562.81A'}
    fit.posterior.model_galaxy.line_fluxes['H  1  6562.81A']


    
def grab_fit_results(fit):

    samples_dict = fit.posterior.samples

    ID = int(fit.galaxy.ID)
    posterior_dict = {'ID': [ID]}
    
    for key, val in samples_dict.items():
        
        if len(val.shape) == 1:
            l16, med, u84 = np.percentiles = np.percentile(val, [16, 50, 84])

            posterior_dict[f'l16_{key}'] = [l16]
            posterior_dict[f'med_{key}'] = [med]
            posterior_dict[f'u84_{key}'] = [u84]


    tab = Table(posterior_dict)

    return tab

def convert_photom_flambda_fnu(fit):

    phot_wave = grab_effective_wavelengths(fit)
    photom_flambda = fit.posterior.samples['photometry']

    photom_fnu = photom_flambda * phot_wave**2 / (2.998e18) #erg/s/cm²/Hz
    photom_fnu_muJy = photom_fnu * 1e29  # convert to microJansky 

    return photom_fnu_muJy

def grab_relevant_photometry(fit, snr_threshold=3):

    ID = int(fit.galaxy.ID)
    data_phot = load_phot(ID)
    chi2 = grab_chi2(fit)

    idx_min_chi2 = np.argmin(chi2)

    photom_data = data_phot[:, 0]
    photom_err = data_phot[:, 1]

    SNR =photom_data / photom_err

    good_data = SNR > snr_threshold

    photom = convert_photom_flambda_fnu(fit)

    phot_l16, phot_med, phot_u84 = np.percentile(photom, [16, 50, 84], axis=0)
    min_photom = photom[idx_min_chi2, :]
    good_min_photom = min_photom[good_data]

    good_phot_med = phot_med[good_data]
    good_phot_data = photom_data[good_data]
    good_phot_err = photom_err[good_data]

    chi2_med_phot = (good_phot_med - good_phot_data) / good_phot_err
    chi2_med_phot = chi2_med_phot**2
    chi2_med_phot = np.sum(chi2_med_phot)

    chi2_min_phot = (good_min_photom - good_phot_data) / good_phot_err
    chi2_min_phot = chi2_min_phot**2
    chi2_min_phot = np.sum(chi2_min_phot)

    num_good_data = np.sum(good_data)

    return chi2_med_phot, chi2_min_phot, num_good_data

def summary_dist_tab(fit, tab, ID):

    chi2_med_phot, chi2_min_phot, num_good_data = grab_relevant_photometry(fit)
    posterior_dict = {'ID': [ID], 
                       'num_good_data': [num_good_data],
                       'chi2_med_phot': [chi2_med_phot],
                       'chi2_min_phot': [chi2_min_phot]}

    columns = tab.colnames
    
    min_chi2_idx = np.argmin(tab['chisq_phot'])
    min_chi2 = np.amin(tab['chisq_phot'])
    posterior_dict['min_chi2'] = [min_chi2]

    for col in columns:
        
        
        u2pt5, l16, med, u84, u97pt5 = np.percentiles = np.percentile(tab[col], [2.5, 16, 50, 84, 97.5])

        posterior_dict[f'l2pt5_{col}'] = [u2pt5]
        posterior_dict[f'l16_{col}'] = [l16]
        posterior_dict[f'med_{col}'] = [med]
        posterior_dict[f'u84_{col}'] = [u84]
        posterior_dict[f'u97pt5_{col}'] = [u97pt5]
        posterior_dict[f'best_{col}'] = [tab[col].data[min_chi2_idx]]  # best value is the one with the minimum chi-squared

    tab = Table(posterior_dict)

    return tab

def grab_sfh_ages(fit):

    ages = fit.posterior.sfh.ages

    return ages

def grab_age_bins(fit):

    age_width = fit.posterior.sfh.age_widths
    return age_width


def grab_2d_sfh(fit):

    sfh_2d = fit.posterior.samples['sfh']

    return sfh_2d

def grab_photometry(fit):

    photometry = fit.posterior.samples['photometry']

    return photometry

def grab_effective_wavelengths(fit):
    
    phot_wave = fit.fitted_model.galaxy.filter_set.eff_wavs

    return phot_wave

def grab_seds(fit):
    
    sed = fit.posterior.samples['spectrum_full']

    return sed

def grab_sed_wave(fit):

    return fit.posterior.model_galaxy.wavelengths

def grab_redshift(fit):

    if 'redshift' in fit.posterior.samples.keys():
        return fit.posterior.samples['redshift']
    else:
        return fit.fit_instructions['redshift']

def grab_observed_wavelengths(fit):

    z = grab_redshift(fit)
    sed_wave = grab_sed_wave(fit)

    #if isinstance(z, float):
    observed_wave = sed_wave * (1 + z)
    #else:
    #    observed_wave = sed_wave[:, np.newaxis] * (1 + z.values[:, np.newaxis])

    return observed_wave

def grab_chi2(fit):
    """Grabs the chi-squared value from the fit object.
    Args:
        fit (pipes.fit): The fit object containing the chi-squared value.
    Returns:
        float: The chi-squared value from the fit object.
    """
    if 'chisq_phot' in fit.posterior.samples.keys():
        chi2 = fit.posterior.samples['chisq_phot']
    else:
        fit.posterior.get_advanced_quantities()
        chi2 = fit.posterior.samples['chisq_phot']
    return chi2


def plot_model_sed(fit, kind = 'median', save_path = None):

    """Plots the model SED from the fit object. 
    Args:
        fit (pipes.fit): The fit object containing the model SED.

    Returns:
        None: Displays the plot.
    """
    

    obs_sed_wavelength = grab_observed_wavelengths(fit)
    seds = grab_seds(fit)
    eff_wave_phot = grab_effective_wavelengths(fit)
    photom = grab_photometry(fit)
    ID = int(fit.galaxy.ID)

    data_phot = load_phot(ID)

    data_phot_flux = data_phot[:, 0]
    data_phot_err = data_phot[:, 1]

    #had to add this check to handle any bad data, this is used mainly to catch SHELA photometric data
    bad_data_mask = (data_phot_flux < -90) & (data_phot_err < -90)

    data_phot_flux[bad_data_mask] = 0
    data_phot_err[bad_data_mask] = 1e12  # Set a large error for bad data

    chi2 = grab_chi2(fit)

    redshift = grab_redshift(fit)

    if save_path is None:
        fig, ax = plot_bagpipes_models(obs_sed_wavelength, seds, 
                                        eff_wave_phot, photom, chi2, redshift, 
                                        data_phot_flux, data_phot_err, kind = kind, save_path = save_path)

        return fig, ax 
    else:
        plot_bagpipes_models(obs_sed_wavelength, seds, 
                             eff_wave_phot, photom, chi2, redshift, 
                             data_phot_flux, data_phot_err, kind = kind, save_path = save_path)


def plot_sfh_fit(fit):
    """Plots the star formation history (SFH) from the fit object.
    Args:
        fit (pipes.fit): The fit object containing the SFH data.
    Returns:    
        fig, ax: The figure and axis objects containing the SFH plot.

    """
    ages = grab_sfh_ages(fit)
    sfh = grab_2d_sfh(fit)
    fig, ax = plot_sfh(ages, sfh)

    return fig, ax

def compute_stellar_mass(fit, age_low = 0, age_high = 10e6):

    """Computes the stellar mass from the fit object based on the age of the galaxy.
    Args:
        fit (pipes.fit): The fit object containing the model SED.
        age_low (float): The lower limit of the age range in Myr.
        age_high (float): The upper limit of the age range in Myr.
    Returns:
        float: The computed stellar mass in solar masses.
    """

    age_bins = grab_age_bins(fit)
    sfh_2d = grab_2d_sfh(fit)

    age_mask = (age_bins >= age_low) & (age_bins <= age_high)
    sfh_2d_masked = sfh_2d[:, age_mask]
    
    age_bins_masked = age_bins[age_mask]

    mass = age_bins_masked * sfh_2d_masked

    stellar_masses = np.sum(mass, axis=1)

    return stellar_masses

def compute_sfr(fit, age_low = 0, age_high = 10e6):
    
    """Computes the star formation rate (SFR) from the fit object based on the age of the galaxy.
    Args:
        fit (pipes.fit): The fit object containing the model SED.
        age_low (float): The lower limit of the age range in Myr.
        age_high (float): The upper limit of the age range in Myr.
    Returns:
        float: The computed star formation rate in solar masses per year.
    """ 

    age_bins = grab_age_bins(fit)
    sfh_2d = grab_2d_sfh(fit)

    age_mask = (age_bins >= age_low) & (age_bins <= age_high)
    sfh_2d_masked = sfh_2d[:, age_mask]
    
    avg_sfr = np.mean(sfh_2d_masked, axis=1)

    return avg_sfr

def compute_burstiness(fit, 
                        age_low_recent = 0, age_high_recent = 10e6, 
                        age_low_old = 10e6, age_high_old = 100e6):

    """Computes the burstiness of star formation from the fit object based on the age of the galaxy.
    Args:
        fit (pipes.fit): The fit object containing the model SED.       
        age_low_recent (float): The lower limit of the recent age range in Myr.
        age_high_recent (float): The upper limit of the recent age range in Myr.
        age_low_old (float): The lower limit of the old age range in Myr.
        age_high_old (float): The upper limit of the old age range in Myr.
    Returns:
        float: The computed burstiness value.
    """ 

    sfr_recent = compute_sfr(fit, age_low_recent, age_high_recent)
    sfr_old = compute_sfr(fit, age_low_old, age_high_old)
    burstiness = sfr_recent / sfr_old

    # Ensure burstiness is not NaN or infinite

    burstiness = np.log10(burstiness)

    return burstiness


def compute_M1500_from_flux(wavelength_rest, Fnu_rest, redshift):
    """

    M1500 = −2.5 log (f0)−48.6−5 log (dL/10)+2.5 log (1 + zp), from: https://www.aanda.org/articles/aa/pdf/2020/06/aa37340-19.pdf
    
    Compute M1500 from Bagpipes Fnu SED.

    Parameters:
    - wavelength_rest: array of rest-frame wavelengths (Å)
    - Fnu_rest: array of rest-frame flux densities (µJy)
    - redshift: galaxy redshift

    Returns:
    - M1500: absolute magnitude at 1500 Å (AB)
    """

    Om0 = .315
    Ode0 = 1 - Om0
    cosmo = LambdaCDM(H0 = 67.4, 
                      Om0 = .315, 
                      Ode0 = Ode0)
    
    # Interpolate Fnu at 1500 Å
    Fnu_1500_Fnu = np.interp(1500, wavelength_rest, Fnu_rest)  # Fnu
    Fnu_1500_jy = Fnu_1500_Fnu * 1e23 # convert to Jy

    # Apparent AB magnitude
    m1500 = -2.5 * np.log10(Fnu_1500_jy / 3631)

    # Luminosity distance in parsecs
    Dl_pc = cosmo.luminosity_distance(redshift).to(u.pc).value

    # Distance modulus
    mu = 5 * np.log10(Dl_pc / 10)

    # Absolute magnitude with K-correction (-2.5 log(1 + z))
    M1500 = m1500 - mu + 2.5 * np.log10(1 + redshift)

    return M1500

def spec_wavelength(fit):

    spec_wave = fit.posterior.model_galaxy.wavelengths

    return spec_wave

def flambda_to_fnu(wavelength, flambda):
    """
    Convert F_lambda [erg/s/cm²/Å] to F_nu [erg/s/cm²/Hz].

    Parameters:
    - wavelength_angstrom: array or scalar, wavelength in Ångstroms
    - flambda_erg_s_cm2_A: array or scalar, F_lambda in erg/s/cm²/Å

    Returns:
    - fnu_erg_s_cm2_Hz: array or scalar, F_nu in erg/s/cm²/Hz
    """
    c_A_s = 2.99792458e18  # speed of light in A/s

    fnu = flambda * wavelength**2 / c_A_s

    return fnu

def compute_m1500(fit, redshift):
    
    wave = spec_wavelength(fit) #rest frame wavelengths
    seds_flambda = fit.posterior.samples['spectrum_full'] #observed spectrum
    seds_m1500 = []
    
    for sed in tqdm(seds_flambda, desc="Computing M1500"):
        fnu = flambda_to_fnu(wave*(1+redshift), sed) #observed fnu
        fnu_rest = fnu*(1+redshift) #rest frame fnu

        m1500 = compute_M1500_from_flux(wave, fnu, redshift)
        seds_m1500.append(m1500)

    return seds_m1500

def compute_lyman_alpha_continuum(fit, continuum_type = 'observed'):

    """ Computes the Lyman-alpha continuum from the fit object.
    Args:   
        fit (pipes.fit): The fit object containing the model SED.

    Returns:
        float: The computed Lyman-alpha continuum in erg/s/cm²/A.
    """
    z = grab_redshift(fit)
    rest_sed_wave = grab_sed_wave(fit)
    obs_seds = grab_seds(fit)
    
    lya_cont_range = np.linspace(1220, 1235, 50)

    continuum_fits = []

    if continuum_type == 'observed':
        
        # Convert rest-frame wavelengths to observed wavelengths
        for sed in obs_seds:
            continnum = np.interp(lya_cont_range, rest_sed_wave, sed)
            continuum_fits.append(np.mean(continnum))

    elif continuum_type == 'rest':

        for sed in obs_seds:
            continnum = np.interp(lya_cont_range, rest_sed, sed * (1 + z))
            continuum_fits.append(np.mean(continnum))
        
    else:
        raise ValueError("Invalid continuum_type. Choose 'observed' or 'rest'.")
    continuum_fits = np.array(continuum_fits)

    return continuum_fits

def calculate_beta_and_muv(fit):
    try:
        fit.posterior.samples['spectrum_full']
    except:
        fit.posterior.get_advanced_quantities()

    # compute Muv and beta from the posterior spectra
    try:
        z = fit.posterior.samples['redshift']

    except:
        z = fit.fit_instructions['redshift']

    Om0 = .315
    Ode0 = 1 - Om0
    cosmo = LambdaCDM(H0 = 67.4, 
                        Om0 = .315, 
                        Ode0 = Ode0)

        
    flam = fit.posterior.samples['spectrum_full']
    N = len(flam)

    lam_rest = fit.posterior.model_galaxy.wavelengths/1e4  #converts it to microns


    if isinstance(z, (np.ndarray)):
        lam_obs = lam_rest * (1+z.reshape(-1, 1))
    else:
        
        lam_obs = lam_rest * (1+z)
    c = 2.998e8
    fnu = flam * lam_obs**2/c * 1e21 # in Jy

    # calzetti 1994 wavelength windows 
    windows = ((lam_rest>=.1268)&(lam_rest<=.1284))|((lam_rest>=.1309)&(lam_rest<=.1316))|((lam_rest>=.1342)&(lam_rest<=.1371))|((lam_rest>=.1407)&(lam_rest<=.1515))|((lam_rest>=.1562)&(lam_rest<=.1583))|((lam_rest>=.1677)&(lam_rest<=.1740))|((lam_rest>=.1760)&(lam_rest<=.1833))|((lam_rest>=.1866)&(lam_rest<=.1890))|((lam_rest>=.1930)&(lam_rest<=.1950))|((lam_rest>=.2400)&(lam_rest<=.2580))
    beta = np.zeros(N)
    for i in range(N):
        fl = flam[i,:]
        p = np.polyfit(np.log10(lam_rest[windows]), np.log10(fl[windows]), deg=1)
        beta[i] = p[0]

    MUV = compute_m1500(fit, z)

    return np.array(beta), MUV


def merged_1d_posterior(fit):

    tab = grab_1d_data(fit)
    beta, muv = calculate_beta_and_muv(fit)
    burstiness = compute_burstiness(fit)
    continuum = compute_lyman_alpha_continuum(fit, continuum_type='observed')
    #ID = int(fit.galaxy.ID)

    tab['beta'] = beta
    tab['Muv'] = muv
    tab['burstiness'] = burstiness
    tab['lya_continuum'] = continuum
    #tab['gal_ID'] = f'{field_name}_{ID}'

    return tab

def grab_posterior_files(output_dir):

    files = [x for x in glob.glob(f'{output_dir}/*.fits.gz')]
    return files

def load_posterior_tables(files):

    tabs = [Table.read(file) for file in files]

    return tabs

def merge_posterior_tables(tabs):

    merged_tabs = vstack(tabs)

    return merged_tabs

def main_merge(output_dir='summary_output'):
    
    """ Merges all posterior tables in the specified directory into a single table. 
    Args:
        output_dir (str): The directory containing the posterior tables to merge.
    Returns:
        astropy.table.Table: The merged table containing all posterior samples.
    """ 

    files = grab_posterior_files(output_dir)
    tabs = load_posterior_tables(files)
    merged_tab = merge_posterior_tables(tabs)
    return merged_tab

def add_hdr_flux_to_posterior(post_tab, flux, flux_err, zlya):

    perturbed_fluxes = np.random.normal(loc = flux, scale = flux_err, size = len(post_tab))

    EW = perturbed_fluxes / post_tab['lya_continuum']
    
    EW_r = EW/(1+zlya)

    post_tab['perturbed_flux'] = perturbed_fluxes
    post_tab['EW_lya'] = EW
    post_tab['EW_lya_rest'] = EW_r

    return post_tab 

def load_hdr_table(file):
    
    cols = ['New_IDs', 'flux', 'flux_err', 'lya_z']

    hdr_tab = Table.read(file)
    hdr_df = hdr_tab.to_pandas()

    hdr_df = hdr_df[cols].set_index('New_IDs')

    return hdr_df 

def main(fit):
    
    hdr_tab = load_hdr_table(f'Matched_Catalogs/{field_name}_matched_hdr_df.fits')
    ID = int(fit.galaxy.ID)

    if not os.path.exists(f'summary_output/Summary_{field_name}_{ID}.fits'):
        
        print(f'No summary file for {field_name} {ID}, generating one.')
        merged_tab = merged_1d_posterior(fit)
        summary_tab = summary_dist_tab(fit, merged_tab, ID)
        
        summary_tab.write(f'summary_output/Summary_{field_name}_{ID}.fits', overwrite=True)
        
        plot_model_sed(fit, save_path = f'sed_plots/SED_{field_name}_{ID}.png')

        reduced_tab = subselect_plot_tab(merged_tab)

        fig = corner_plot(reduced_tab)
        fig.savefig(f'summary_plots/Corner_{field_name}_{ID}.png', dpi=150)

        plt.close('all')

        rename_cols = {'continuity:dsfr1': 'dsfr1',
                    'continuity:dsfr2': 'dsfr2',
                    'continuity:dsfr3': 'dsfr3',
                    'continuity:dsfr4': 'dsfr4',
                    'continuity:dsfr5': 'dsfr5',
                    'continuity:dsfr6': 'dsfr6',
                    'continuity:dsfr7': 'dsfr7',
                    'continuity:massformed': 'massformed',
                    'continuity:metallicity': 'metallicity',
                    'dust:Av': 'Av',
                    'dust:B': 'B',
                    'dust:delta': 'delta',
                    'nebular:logU': 'logU'}

        for old_col, new_col in rename_cols.items():
            
            merged_tab.rename_column(old_col, new_col)

        merged_tab['gal_ID'] = f'{field_name}_{ID}'

        flux = hdr_tab.loc[ID, 'flux'] * 1e-17  # Convert to erg/s/cm²
        flux_err = hdr_tab.loc[ID, 'flux_err'] * 1e-17  # Convert to erg/s/cm²
        zlya = hdr_tab.loc[ID, 'lya_z']

        merged_tab = add_hdr_flux_to_posterior(merged_tab, flux, flux_err, zlya)
        
        merged_tab.write(f'summary_output/Posterior_Distribution_{field_name}_{ID}.fits.gz', overwrite=True)
    
    else:
        print(f'Summary file for {field_name} {ID} already exists, skipping.')


def process_one_fit(x):
    fit = grab_bagpipes_output(x)
    main(fit)
    del fit

def find_fitted_ids():
    import os 
    files = [x for x in glob.glob(f'pipes/posterior/{field_name}/*.h5')]
    fitted_ids = [int(os.path.basename(file).split('.')[0]) for file in files]
    return fitted_ids

def run_serial(fitted_ids):
    """ Runs the main function serially for a list of fitted IDs.
    Args:
        fitted_ids (list): List of IDs for which to run the main function.
    Returns:    
        None: The function runs the main function for each ID in the list.
    """
    for ID in tqdm(fitted_ids, desc="Processing IDs"):
        try:
            process_one_fit(ID)
        except Exception as e:
            print(f"Error processing ID {ID}: {e}")
    

def run_parallel(fitted_ids, max_workers=10):
    """
    Runs the main function in parallel for a list of fitted IDs.
    Args:       
        fitted_ids (list): List of IDs for which to run the main function.
        max_workers (int): Maximum number of parallel workers to use. Tested 20 and that seemed to work the best as we got no Memory Allocation Error 
    Returns:
        None: The function runs the main function in parallel for each ID.
    """


    from concurrent.futures import ProcessPoolExecutor
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_one_fit, fitted_ids), total=len(fitted_ids)))

    except KeyboardInterrupt:
        print("Interrupted! Cancelling all futures...")
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise  # re-raise the exception to stop the script

def merge_summary_by_ids():

    IDS = Bagpipes_Phot_tab['New_IDs'].data

    tabs = []

    for ids in tqdm(IDS, desc="Processing IDs"):

        try:
            t = Table.read(f'summary_output/Summary_{field_name}_{ids}.fits')
            tabs.append(t)
        except:

            table=tabs[0]
            tab_cols = table.colnames
            t = Table({x: [np.nan] for x in tab_cols})
            t['ID'] = ids
            tabs.append(t)

    merged_tab = vstack(tabs)

    columns = merged_tab.colnames

    for col in columns:
        new_name = col.replace(':', '_')
        try:
            merged_tab.rename_column(col, new_name)
        except KeyError:
            print(f"skipping {col} as it is already renamed")

    return merged_tab

def merge_test():

    IDS = Bagpipes_Phot_tab['New_IDs'].data

    tabs = []

    for ids in tqdm(IDS[:10], desc="Processing IDs"):

        try:
            t = Table.read(f'summary_output/Summary_{field_name}_{ids}.fits')
            tabs.append(t)
        except:

            pass

    merged_tab = vstack(tabs)

    columns = merged_tab.colnames

    for col in columns:
        new_name = col.replace(':', '_')
        try:
            merged_tab.rename_column(col, new_name)
        except KeyError:
            print(f"skipping {col} as it is already renamed")

    return merged_tab

def make_ML_table(tab):

    cols = ['massformed',
            'metallicity',
            'Av',
            'B',
            'delta',
            'logU',
            'stellar_mass',
            'sfr',
            'ssfr',
            'mass_weighted_age',
            'mass_weighted_zmet',
            'beta',
            'Muv',
            'burstiness',
            'lya_continuum',
            'perturbed_flux',
            'EW_lya',
            'EW_lya_rest', 
             'gal_ID',
             'chisq_phot']

    ml_tab = tab[cols]

    return ml_tab

def merge_all_posteriors(IDs, output_dir='summary_output'):
    """
    Merges all posterior tables in the specified directory into a single table.
    Args:
        output_dir (str): The directory containing the posterior tables to merge.
    Returns:
        astropy.table.Table: The merged table containing all posterior samples.
    """
    #files = grab_posterior_files(output_dir)
    files = [f'{output_dir}/Posterior_Distribution_{field_name}_{ID}.fits.gz' for ID in IDs]
    tabs = load_posterior_tables(files)
    merged_tab = merge_posterior_tables(tabs)
    merged_tab.write(f'{output_dir}/Merged_Posterior_Distribution_{field_name}.fits.gz', overwrite=True)
    ml_tab = make_ML_table(merged_tab)
    ml_tab.write(f'{output_dir}/Merged_ML_Table_{field_name}.fits.gz', overwrite=True)


def generate_merge_ml_tables(fitted_ids, output_dir='summary_output'):

    files = []

    for ID in fitted_ids:
        if os.path.exists(f'{output_dir}/Posterior_Distribution_{field_name}_{ID}.fits.gz'):
            files.append(f'{output_dir}/Posterior_Distribution_{field_name}_{ID}.fits.gz')

    tabs = [Table.read(file) for file in files]
    ml_tabs = [make_ML_table(tab) for tab in tabs]
    merged_tab = vstack(ml_tabs)
    
    return merged_tab

def find_remaining_ids():

    All_IDs = Bagpipes_Phot_tab['New_IDs'].data
    fitted_ids = find_fitted_ids()

    remaining_ids = set(All_IDs) - set(fitted_ids)

    return remaining_ids


def quick_lookup_summary(ID):

    df = Table.read('summary_output/Merged_Summary_{field_name}.fits').to_pandas().set_index('New_IDs')
    return df.loc[ID]




def bring_up_single_ID_stats(ID):
    output = 'quick_look'
    try:
        sed_plot_file = 'sed_plots/SED_{field_name}_{ID}.png'
        corner_plot_file = 'summary_plots/Corner_{field_name}_{ID}.png'

        img_sed = img = mpimg.imread(sed_plot_file)
        plt.imshow(img)
        plt.axis('off')  # optional, hides axis
        plt.savefig(f'{output}/SED_{field_name}_{ID}.png', dpi=150)

        img_corner = mpimg.imread(corner_plot_file)
        plt.imshow(img_corner)
        plt.axis('off')  # optional, hides axis
        #plt.show()
        plt.savefig(f'{output}/Corner_{field_name}_{ID}.png', dpi=150)

        

    except FileNotFoundError:
        fit = grab_bagpipes_output(ID)

        fig, ax = plot_model_sed(fit)
        fig.savefig(f'{output}/SED_{field_name}_{ID}.png', dpi=150)

        merged_tab = merged_1d_posterior(fit)
        reduced_tab = subselect_plot_tab(merged_tab)
        fig = corner_plot(reduced_tab)
        #plt.show()
        fig.savefig(f'{output}/Corner_{field_name}_{ID}.png', dpi=150)


    fig, ax = plot_sfh_fit(fit)
    ax.set_xlim(0, 200)
    #plt.show()
    fig.savefig(f'{output}/SFH_{field_name}_{ID}.png', dpi=150)


def map_index_to_new_ID(index):

    bp_df = Bagpipes_Phot_tab.to_pandas()
    bp_df = bp_df.set_index('ID')
    new_IDs  = bp_df.loc[index, 'New_IDs']

    return new_IDs

if __name__ == "__main__":

    test = args.test
    merge_summary = args.merge_summary
    merge_posteriors = args.merge_posteriors
    test_merge = args.test_merge

    if test:
        test_functions()
        exit()

    if test_merge:
        
        tab = merge_test()
        print(tab)
        print("Test Merge Completed")
        exit()

    if merge_summary:
        print("Merging Summary Tables")
        merged_tab = merge_summary_by_ids()
        merged_tab.write(f'summary_output/Merged_Summary_{field_name}.fits', overwrite=True)
        exit()

    if merge_posteriors:
        print("Merging Posterior Tables")
        IDs = Bagpipes_Phot_tab['New_IDs'].data
        merged_tab = merge_all_posteriors(IDs)
        #merged_tab.write(f'summary_output/Merged_Posterior_Distribution_{field_name}.fits.gz', overwrite=True)
        exit()

    else:
        index = args.id

        #ALL_IDs = Bagpipes_Phot_tab['New_IDs'].data
        fitted_ids = find_fitted_ids()

        gal_ID = fitted_ids[index]

        print(f'Processing Index: {index}, Galaxy ID: {gal_ID}', flush=True)

        process_one_fit(gal_ID)
    