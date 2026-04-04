from config_loader import open_config, get_field_info, grab_min_and_max_filt_wavelengths
from file_io import read_astropy_table
from bp_utils import fit_bp, make_lya_mask, quality_check_photom, convert_table_to_DF, get_redshift, Galaxy_Model_Builder, fit_instruction_nebular_fixedz
from refactored_post_analysis import main_summary
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run BAGPIPES SED fitting and post-fit analysis for a specific field.")
    parser.add_argument('--phot_config', type=str, required=True, help='config file for the photometry')
    parser.add_argument('--bp_config', type=str, required=True, help='config file for the bp model')
    parser.add_argument('--field', type=str, required=True, help='Field name to use (e.g., CEERS, COSMOS, TONE, SHELA, UDS, or GOODSN)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run')
    parser.add_argument('-id', type=int, required=True, help='Object index to process')

    # Output path overrides — fall back to values in config if not provided
    parser.add_argument('--summary_dir', type=str, default=None, help='Output dir for Summary FITS files (overrides config)')
    parser.add_argument('--sed_plots_dir', type=str, default=None, help='Output dir for SED plots (overrides config)')
    parser.add_argument('--corner_plots_dir', type=str, default=None, help='Output dir for corner plots (overrides config)')
    parser.add_argument('--posterior_dir', type=str, default=None, help='Output dir for posterior distribution files (overrides config)')
    parser.add_argument('--hdr_catalog_dir', type=str, default=None, help='Dir containing matched HDR catalogs (overrides config)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files instead of skipping')

    args = parser.parse_args()

    FILE = args.phot_config
    param_file = args.bp_config
    run = args.run_name

    config = open_config(FILE)
    param_config = open_config(param_file)
    field_name = args.field

    filters, phot_file, flux_cols, fluxerr_cols, bp_sf_model = get_field_info(config, field_name)
    filter_min, filter_max = grab_min_and_max_filt_wavelengths(config, field_name)

    # Build output_dirs: CLI args take priority, config output section is the default
    config_output = config.get('output', {})
    output_dirs = {
        'summary_dir':      args.summary_dir      or config_output.get('summary_dir',      'summary_output'),
        'sed_plots_dir':    args.sed_plots_dir     or config_output.get('sed_plots_dir',    'sed_plots'),
        'corner_plots_dir': args.corner_plots_dir  or config_output.get('corner_plots_dir', 'summary_plots'),
        'posterior_dir':    args.posterior_dir     or config_output.get('posterior_dir',    'summary_output'),
        'hdr_catalog_dir':  args.hdr_catalog_dir   or config_output.get('hdr_catalog_dir',  'Matched_Catalogs'),
    }

    Bagpipes_Phot_tab = read_astropy_table(phot_file)
    Bagpipes_Phot_DF = convert_table_to_DF(Bagpipes_Phot_tab)

    def load_phot(ID):

        ID = int(ID)

        zspec = get_redshift(Bagpipes_Phot_DF, ID)
        lya_mask = make_lya_mask(filter_min, filter_max, zspec)

        photom_flux = Bagpipes_Phot_DF.loc[ID, flux_cols].values
        photom_flux_err = Bagpipes_Phot_DF.loc[ID, fluxerr_cols].values

        photom_flux[lya_mask] = 0
        photom_flux_err[lya_mask] = 1e10

        photom_flux, photom_flux_err = quality_check_photom(photom_flux, photom_flux_err, flux_cols)

        return np.c_[photom_flux.astype(float), photom_flux_err.astype(float)]

    def fit_BP(index, filters, load_func, z, run, model='bursty'):

        BP_Galaxy = Galaxy_Model_Builder(index, load_func, filters)
        fit_instructions = fit_instruction_nebular_fixedz(z, model, {'bp_sf_model': bp_sf_model})
        fit = fit_bp(BP_Galaxy, fit_instructions, run)

        return fit

    IDs = Bagpipes_Phot_DF.index.values
    ID = IDs[args.id]

    z = get_redshift(Bagpipes_Phot_DF, ID)
    fit = fit_BP(ID, filters, load_phot, z, run)

    main_summary(fit, field_name, load_phot, output_dirs, overwrite=args.overwrite)




