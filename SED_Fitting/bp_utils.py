import bagpipes as pipes
import numpy as np
from astropy.cosmology import Planck18 as cosmo


def convert_table_to_DF(tab):
   DF = tab.to_pandas().set_index('New_IDs')
   return DF

def get_redshift(DF, ID):
    z_tesla = DF.loc[ID, 'Redshift']
    return z_tesla


def Galaxy_Model_Builder(ID, load_func, filters, spectrum = False):
    '''
    This function will make a bagpipes galaxy model for one ID
    
    '''
    #need to take the extra step of converting ID to string for bagpipes galaxy function to work properly
    galaxy_bp = pipes.galaxy(str(ID), load_func, filt_list = filters, spectrum_exists=spectrum)
    
    return galaxy_bp

def make_non_param_instructions(config, z):

    #values taken from the PLANCK CMB 2018 paper
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

def make_delayed_tau_instructions(config, z):
    
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

def make_bursty_instructions(config, z):

    #values taken from the PLANCK CMB 2018 paper    
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

def fit_instruction_nebular_fixedz(z, model, config):
    
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
        
        fit_instructions = make_delayed_tau_instructions(config, z)
    
    elif model == 'nonparam':
        #values taken from the PLANCK CMB 2018 paper
        fit_instructions = make_non_param_instructions(config, z)
        
    elif model == "bursty":
        
        #values taken from the PLANCK CMB 2018 paper
        fit_instructions = make_bursty_instructions(config, z)
    
    return fit_instructions

def make_lya_mask(filter_min, filter_max, zspec):

    lya = 1215.67
    obs_lya = lya * (1 + zspec)
    lya_mask = (filter_min < obs_lya) & ( obs_lya < filter_max)

    return lya_mask

def quality_check_photom(photom_flux, photom_flux_err, flux_cols, bad_snr = -3, max_SNR = 20):

    #getting the snr of sources
    snr = photom_flux/photom_flux_err
    
    #if the snr is below -5 then we know it is bad we make the flux 0 and error really big
    bad_flux_idx = snr < bad_snr

    #making a special mask for the IRAC bands
    irac_mask = np.array([('ch1' in x) | ('ch2' in x) for x in flux_cols])

    #adding in a 5 % error to the fluxes
    photom_flux_err[~irac_mask] = np.sqrt(photom_flux_err[~irac_mask]**2 + (0.05 * photom_flux[~irac_mask])**2) # we add a 5% error from the fluxes capping the SNR to be at max 20
    
    #adding in a 20% error on the irac fluxes 
    photom_flux_err[irac_mask] = np.sqrt(photom_flux_err[irac_mask]**2 + (0.2 * photom_flux[irac_mask])**2)     # we add a 20% error from the fluxes capping the SNR to be at max 10

    #setting bad flux to a really small value and error to be really big
    photom_flux[bad_flux_idx] = 0
    photom_flux_err[bad_flux_idx] = 1e12

    check_snr = photom_flux/photom_flux_err

    high_SNR_mask = check_snr > max_SNR

    photom_flux_err[high_SNR_mask] = photom_flux[high_SNR_mask]/max_SNR

    return photom_flux, photom_flux_err

def fit_bp(BP_Galaxy, fit_instructions, run):
    
    fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    
    fit.fit(verbose=True)

    return fit