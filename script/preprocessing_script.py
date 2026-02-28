import numpy as np
from astropy.table import Table

def generate_lyaz(obs_wave, wave_err):
    
    lya_rest = 1215.67
    # obs_wave = hdr_df.loc[ids, 'wave'].values
    # wave_err = hdr_df.loc[ids, 'wave_err'].values
    
    ratio = obs_wave/lya_rest
    ratio_err = np.abs(ratio)*(wave_err/obs_wave)
    
    lyaz = (ratio) - 1
    
    return lyaz, ratio_err

def DF_to_astropy_table(df):
    return Table.from_pandas(df)


def convert_photom_flux(phot_catalog, flux_cols, fluxerr_cols, units):

    if units == 'nJy':
        
        for f, ferr in zip(flux_cols, fluxerr_cols):
        
            phot_catalog[f] = phot_catalog[f].data/1000 #converting to muJy
            phot_catalog[ferr] = phot_catalog[ferr].data/1000 #converting to muJy

    return phot_catalog
    

def photom_quality_check(phot_catalog, flux_cols, fluxerr_cols, 
                         snr_threshold = -3, 
                         default_flux = 0, default_error = 1e12):

    for f, ferr in zip(flux_cols, fluxerr_cols):
        
        signal = phot_catalog[flux_cols]
        noise  = phot_catalog[fluxerr_cols]

        SNR = signal/noise

        snr_mask = SNR < snr_threshold
        nan_mask = ~np.isfinite(noise) | ~np.isfinite(signal)
        zero_error_mask = noise == 0

        mask = snr_mask | nan_mask | zero_error_mask

        signal[mask] = default_flux
        noise[mask] = default_error

        phot_catalog[flux_cols] = signal
        phot_catalog[fluxerr_cols] = noise

    return phot_catalog




