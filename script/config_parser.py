
def grab_useful_columns(config, field_name):
    
    """ Function to grab the useful columns from the config file for a given field. 
    Parameters
    ----------
    field_name : str

        The name of the field for which to grab the useful columns.

    Returns
    -------
    list
        A list of useful columns for the specified field.
    """
    try:
        field_data = config[field_name]
        useful_columns = field_data['useful_columns']
        return useful_columns
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")

def grab_flux_columns(config, field_name):
    """ Function to grab the flux columns from the config file for a given field.
    Parameters
    ----------
    field_name : str
        The name of the field for which to grab the flux columns.
    Returns
    -------
    tuple
        A tuple containing the flux columns and their corresponding error columns for the specified field.
    """
    try:
        field_data = config[field_name]
        flux_cols = field_data['flux_columns']
        fluxerr_cols = field_data['fluxerr_columns']
        return flux_cols, fluxerr_cols
    except KeyError:
        raise ValueError(f"Field '{field_name}' not found in config")