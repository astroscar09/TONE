from coordinates import cross_match
from file_io import read_config_yaml, read_HDR5, read_in_photom_cat, save_table
from preprocessing_script import generate_lyaz, convert_photom_flux, photom_quality_check, DF_to_astropy_table
from config_parser import grab_useful_columns, grab_flux_columns, 
import argparse

def main(photom_config, selection_config, field, HDR): 
    
    """
    Main function to read in the photometric catalog, HDR data, and match them.
    Parameters:
        photom_file (str): Path to the photometric catalog file.
        field (str): Name of the field to match.
        search_radius (float): Search radius in arcseconds for matching.
        hdr_version (str): Version of the HDR data to use.
    """
    
    if HDR == 'HDR5':

        HDR_DF = read_HDR5(field, selection_config)

    elif HDR == 'HDR4':

        HDR_DF = read_HDR4(field, selection_config)


    phot_file = photom_config[field]['photometry']

    # Read in the photometric catalog
    photom_tab = read_in_photom_cat(phot_file)

    matched_phot, matched_hdr_df = cross_match(photom_tab, HDR_DF, photom_config, selection_config)
    
    cols = grab_useful_columns(photom_config, field)
    flux_cols, fluxerr_cols = grab_flux_columns(photom_config, field) 
    units = photom_config[field]['phot_unit']
    matched_phot_muJy = convert_photom_flux(matched_phot, flux_cols, fluxerr_cols, units)
    good_matched_phot_muJy = photom_quality_check(matched_phot_muJy, flux_cols, fluxerr_cols)
    
    lya = generate_lyaz(matched_hdr_df['wave'].values, matched_hdr_df['wave_err'].values)

    photom_cat['Redshift'] = lya

    mask = lya > 0

    final_phot = good_matched_phot_muJy[mask][cols]
    final_hdr_df = matched_hdr_df[mask]


    return final_phot, final_hdr_df


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Running Cross Match Between Photometric and Spectroscopic Caalogs")
    parser.add_argument('--selection_config', type=str, required=True)
    parser.add_argument('--phot_config', type=str, required=True)  
    parser.add_argument('--field', type=str, required=True)  
    parser.add_argument('--HDR', type=str, required=True)   
    
    args = parser.parse_args()

    config_phot_file      = args.phot_config
    config_selection_file = args.selection_config
    field                 = args.field
    HDR                   = args.HDR

    phot_config = read_config_yaml(config_phot_file)
    selection_config = read_config_yaml(config_selection_file)
    
    photom_cat, matched_df = main(config_phot_file, config_selection_file, field, HDR)

    # Save the matched photometric catalog to a CSV file
    #photom_cat.write(f'Matched_Catalogs/{field}_matched_photom_cat.fits', overwrite = True)
    #Table.from_pandas(matched_df).to_csv(f'Matched_Catalogs/{field}_matched_hdr_df.fits')
    outfile = phot_config[field]['output']
    save_table(photom_cat, outfile)

    hdr_tab = DF_to_astropy_table(matched_df)
    save_table(photom_cat, outfile)
    #save_df()




# 


