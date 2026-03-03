
import numpy as np
import pandas as pd
import bagpipes as pipes
from astropy.cosmology import LambdaCDM
from astropy.table import Table
from SED_Fitting.plotting_utils import *
from tqdm import tqdm
from astropy.table import vstack
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#####
#this script will take the bagpipes output and perform some posterior analysis
#####

def grab_1d_data(fit):
    
    samples_dict = fit.posterior.samples

    posterior_1d = {}
    for key, val in samples_dict.items():
        
        if len(val.shape) == 1:

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

def grab_relevant_photometry(fit, load_phot, snr_threshold=3):

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


def plot_model_sed(fit, load_phot, kind = 'median', save_path = None):

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
            continnum = np.interp(lya_cont_range, rest_sed_wave, sed * (1 + z))
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

    tab['beta'] = beta
    tab['Muv'] = muv
    tab['burstiness'] = burstiness
    tab['lya_continuum'] = continuum

    return tab

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

def main(fit, field_name):
    
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



    

