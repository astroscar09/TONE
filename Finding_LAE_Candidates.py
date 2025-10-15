from astropy.table import Table
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def snr_per_source(ID,  field):
    
    spectra_fit = Table.read(f'spectral_refit/fit_results_{field}_{ID}.fits')
    diff_percentiles = (np.percentile(spectra_fit['integrated_flux'],  q = 84) - np.percentile(spectra_fit['integrated_flux'], q = 16))/2
    SNR = np.median(spectra_fit['integrated_flux'])/diff_percentiles #np.std(spectra_fit['integrated_flux'])

    return SNR

def snr_per_source_amp(ID,  field):
    
    spectra_fit = Table.read(f'spectral_refit/fit_results_{field}_{ID}.fits')
    diff_percentiles = (np.percentile(spectra_fit['A'],  q = 84) - np.percentile(spectra_fit['A'], q = 16))/2
    SNR = np.median(spectra_fit['A'])/diff_percentiles #np.std(spectra_fit['integrated_flux'])

    return SNR

def compute_new_SNR_amp(ids, field):

    new_snr = []
    for ID in tqdm(ids, total = len(ids)):

        SNR = snr_per_source_amp(ID, field)

        new_snr.append(SNR)

    new_snr = np.array(new_snr)

    return new_snr

def compute_new_SNR(ids, field):

    new_snr = []
    for ID in tqdm(ids, total = len(ids)):

        SNR = snr_per_source(ID, field)

        new_snr.append(SNR)

    new_snr = np.array(new_snr)

    return new_snr

def gaussian(x, amp, mean, stddev, offset):
    return amp * np.exp(-0.5 * ((x - mean) / stddev) ** 2) + offset

def recompute_snr_quadrature(ID, field):

    window = 20
    hdr = Table.read(f'Matched_Catalogs/{field}_matched_hdr_df.fits')

    source_hdr = hdr[hdr['New_IDs'] == ID]

    obs_sn = source_hdr['sn'].data[0]
    obs_wavelength = source_hdr['wave'].data[0]

    spectra = Table.read(f'Matched_Catalogs/{field}_LAE_Spectra.fits.gz')
    spectra = spectra[spectra['New_IDs'] == ID]

    wave = spectra['wavelength'].data[0]
    flux = spectra['flux'].data[0]
    flux_err = spectra['flux_err'].data[0]

    mask = (wave > obs_wavelength - window) & (wave < obs_wavelength + window)
    wave = wave[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]
    
    line_fit = Table.read(f'spectral_refit/fit_results_{field}_{ID}.fits').to_pandas()

    random_indices = np.random.randint(0, len(line_fit), 100)

    random_params = line_fit.iloc[random_indices, :-2]

    new_snr_dist = []

    for i, params in random_params.iterrows():

        amp, mean, stddev, offset = params
        model_flux = gaussian(wave, amp, mean, stddev, offset)

        new_SNR = np.sqrt(np.sum((model_flux/flux_err)**2))
        new_snr_dist.append(new_SNR)

    return new_snr_dist


def compute_new_SNR_quad(ids, field):

    new_snr = []
    for ID in ids:

        SNR = recompute_snr_quadrature(ID, field)

        new_snr.append(SNR)

    new_snr = np.array(new_snr)

    return new_snr


def plot_spectra(ID, field):

    hdr = Table.read(f'Matched_Catalogs/{field}_matched_hdr_df.fits')

    source_hdr = hdr[hdr['New_IDs'] == ID]

    obs_wavelength = source_hdr['wave'].data[0]
    obs_sn = source_hdr['sn'].data[0]

    spectra = Table.read(f'Matched_Catalogs/{field}_LAE_Spectra.fits.gz')
    spectra = spectra[spectra['New_IDs'] == ID]

    wave = spectra['wavelength'].data[0]
    flux = spectra['flux'].data[0]
    flux_err = spectra['flux_err'].data[0]

    new_SNR = snr_per_source(ID,  field)

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10, 5))

    ax1 = axes[0]
    ax2 = axes[1]

    ax1.step(wave, flux, where = 'mid', label='Flux', color = 'gray')
    ax1.errorbar(wave, flux, yerr=flux_err, fmt='none', color = 'black')
    ax1.axvline(obs_wavelength, color='red', linestyle='--', label='Observed Wavelength')

    mask = (wave > obs_wavelength - 100) & (wave < obs_wavelength + 100)
    wave = wave[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]
    
    ax2.step(wave, flux, where = 'mid', color = 'gray', label = f'HDR SN: {obs_sn:.2f}')
    ax2.errorbar(wave, flux, yerr=flux_err, fmt='none', color = 'black', label = f'Integrated SN: {new_SNR:.2f}')
    ax2.axvline(obs_wavelength, color='red', linestyle='--')
    #ax2.set_xlim(obs_wavelength - 100, obs_wavelength + 100)
    ax2.legend()
    fig.savefig('quick_look/spectra_{field}_{ID}.png')


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def reading_fits(file):
    """
    Reads a FITS file and returns the data as an Astropy Table.
    
    Parameters
    ----------
    file : str
        The path to the FITS file.
    
    Returns
    -------
    astropy.table.Table
        The data from the FITS file as an Astropy Table.
    """
    return Table.read(file)

def main(int_file, photom_file, field):

    int_tab = reading_fits(int_file)
    photom_tab = reading_fits(photom_file)

    names = [name for name in photom_tab.colnames if len(photom_tab[name].shape) <= 1]
    photom_tab = photom_tab[names]#.to_pandas()

    try:
        photom_df = photom_tab.to_pandas().set_index('ID')

    except KeyError:
        photom_df = photom_tab.to_pandas().set_index('id')

    #photom_df = photom_df.loc[int_tab['ID'].data]
    photom_df['Lya_Mask'] = int_tab['Lya_gt_0pt7'].data

    continuum_filters = config[field]['continuum_filters']

    flux_cols = continuum_filters[:2]
    fluxerr_cols = continuum_filters[2:]

    SNR = photom_df[flux_cols].values / photom_df[fluxerr_cols].values

    mask1 = (SNR[:, 0] > 3) & (SNR[:, 1] > 3)
    mask2 = (SNR[:, 0] > 5) | (SNR[:, 1] > 5)

    mask = mask1 | mask2

    photom_df['SNR_mask'] = mask

    for i, f_cols in enumerate(flux_cols):
        
        snr_col = f_cols.replace('FLUX', 'SNR')
        photom_df[snr_col] = SNR[:, i]


    #final_sample_lae_photom = photom_df[mask]

    if field != 'TONE':
        
        mag_cols = [x.replace('FLUX', 'MAG') for x in flux_cols]

    else:
        
        mag_cols = [x.replace('flux_ujy', 'MAG') for x in flux_cols]

    mag_df = photom_df[mag_cols].values

    continuum_depths = config[field]['continuum_depths']

    cont_mask = (mag_df[:, 0] > continuum_depths[0]) | (mag_df[:, 1] > continuum_depths[1])

    photom_df['Cont_Depth_Mask'] = ~cont_mask

    photom_df = photom_df.reset_index()

    return photom_df

def removing_duplicate_detectids(df):

    df['dup_detids'] = df.duplicated(subset = 'detectid', keep = False).values

    non_dup_df = df[~df.dup_detids.values]
    dup_df = df[df.dup_detids.values]

    dup_ids = list(set(dup_df.detectid.values))

    dup_df = dup_df.set_index('detectid')

    good_ids = []
    for detids in dup_ids:

        dups = dup_df.loc[detids]

        separation = dups.separation.values
        min_sep_idx = np.argmin(separation)

        min_sep = separation[min_sep_idx]

        delta_sep = separation - min_sep

        check_delta_sep = ((delta_sep > 0.4) & (delta_sep > 0)).any()
        if check_delta_sep:

            dups = dups.iloc[min_sep_idx]
            good_ids.append(dups['New_IDs'])

    dup_df = dup_df.reset_index().set_index('New_IDs')

    good_dups = dup_df.loc[good_ids].reset_index()

    good_dups = good_dups[non_dup_df.columns]

    new_laes = pd.concat((non_dup_df, good_dups))

    return new_laes

def removing_dup_Phot_IDs(df, field):

    if field == 'TONE' or field == 'SHELA':
        df['dup_phot'] = df.duplicated(subset = 'id', keep = False).values
    else:
        df['dup_phot'] = df.duplicated(subset = 'ID', keep = False).values


    non_dup_df = df[~df['dup_phot'].values]
    dup_df = df[df['dup_phot'].values]
    if field == 'TONE' or field == 'SHELA':
        dup_ids = list(set(dup_df.id.values))
    else:
        dup_ids = list(set(dup_df.ID.values))

    try:
        dup_df = dup_df.set_index('ID')
    except:
        dup_df = dup_df.set_index('id')

        
    good_ids = []
    for ids in dup_ids:
        dups = dup_df.loc[ids]
        separation = dups.separation.values
        min_sep_idx = np.argmin(separation)
        min_sep = separation[min_sep_idx]

        delta_sep = separation - min_sep

        check_delta_sep = ((delta_sep > 0.4) & (delta_sep > 0)).any()

        if check_delta_sep:

            dups = dups.iloc[min_sep_idx]
            good_ids.append(dups['New_IDs'])

    dup_df = dup_df.reset_index().set_index('New_IDs')

    good_dups = dup_df.loc[good_ids].reset_index()

    good_dups = good_dups[non_dup_df.columns]

    new_laes = pd.concat((non_dup_df, good_dups))

    return new_laes


def grab_median_params(IDS, field):

    posterior_tables = [Table.read(f'summary_output/Posterior_Distribution_{field}_{x}.fits.gz') for x in IDS]

    # Compute the median of each parameter across all posterior samples
    median_params = {}

    for param in ['Av', 'EW_lya_rest']:
        median_params[param] = [np.median(table[param]) for table in posterior_tables]

    return pd.DataFrame(median_params, index=IDS)

