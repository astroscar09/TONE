import yaml
import numpy as np

def open_config(file):

    with open(file, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_field_info(config, field_name):
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
    
def grab_min_and_max_filt_wavelengths(config, field_name):

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