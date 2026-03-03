from config_loader import open_config, get_field_info, grab_min_and_max_filt_wavelengths
from file_io import read_astropy_table
from bp_utils import fit_bp, make_lya_mask, quality_check_photom, convert_table_to_DF, get_redshift, Galaxy_Model_Builder, fit_instruction_nebular_fixedz
import numpy as np
import argparse

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description="Run BAGPIPES for a specific field.")
    parser.add_argument('--phot_config', type=str, required=True, help='config file for the photometry')
    parser.add_argument('--bp_config', type=str, required=True, help='config file for the bp model')
    parser.add_argument('--field', type=str, required=True, help='Field name to use (e.g., CEERS, COSMOS, TONE, SHELA, UDS, or GOODSN)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run')
    parser.add_argument('-id', type=int, required=True, help='Object index to process')

    args = parser.parse_args()  
    
    FILE = args.phot_config
    param_file = args.bp_config
    run = args.run_name

    config = open_config(FILE)
    param_config = open_config(param_file)
    field_name = args.field

    filters, phot_file, flux_cols, fluxerr_cols, bp_sf_model = get_field_info(config, field_name)
    filter_min, filter_max = grab_min_and_max_filt_wavelengths(config, field_name)

    Bagpipes_Phot_tab = read_astropy_table(phot_file)
    Bagpipes_Phot_DF = convert_table_to_DF(Bagpipes_Phot_tab)


    def load_phot(ID):
        
        ID = int(ID)
        
        zspec = get_redshift(Bagpipes_Phot_DF, ID)

        lya_mask = make_lya_mask(filter_min, filter_max, zspec)

        #getting the full flux and flux error info
        photom_flux = Bagpipes_Phot_DF.loc[ID, flux_cols].values
        photom_flux_err = Bagpipes_Phot_DF.loc[ID, fluxerr_cols].values  

        #making lya bands have 0 flux and large errors
        photom_flux[lya_mask] = 0
        photom_flux_err[lya_mask] = 1e10
        
        photom_flux, photom_flux_err = quality_check_photom(photom_flux, photom_flux_err, flux_cols)

        phot = np.c_[photom_flux.astype(float), photom_flux_err.astype(float)] 

        return phot

    def fit_BP(index, filters, load_func, z, run, model = 'bursty'):
        
        BP_Galaxy = Galaxy_Model_Builder(index, load_func, filters)
        fit_instructions = fit_instruction_nebular_fixedz(z, model = model)
        fit = fit_bp(BP_Galaxy, fit_instructions, run)

    

    IDs = Bagpipes_Phot_DF.index.values
    ID = IDs[args.id]
    
    z = get_redshift(Bagpipes_Phot_DF, ID)
    fit_BP(ID, filters, load_phot, z, run)




