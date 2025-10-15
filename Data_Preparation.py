import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
from tqdm import tqdm 


def load_posterior_distribution(field, ID):

    return Table.read(f'summary_output/Posterior_Distribution_{field}_{ID}.fits.gz')

def all_posteriors(fields, IDS):

    tabs = []
    for IDS, field in tqdm(zip(IDS, fields), total=len(IDS)):
        tabs.append(load_posterior_distribution(field, IDS))

    return tabs

def make_epoch_data(tables):

    ml_cols = ['massformed',
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
                'EW_lya_rest']

    ml_tab = []
    
    for tab in tqdm(tables, total = len(tables), desc='Making ML table'):

        rand_idx = np.random.randint(0, len(tab))
        ml_tab.append(tab[rand_idx])

    ml_tab = vstack(ml_tab)[ml_cols]

    return ml_tab


def pad_table(tab, target_rows = 500, fill_value=0):
    """Pad an astropy Table with rows until it has target_rows."""
    n_missing = target_rows - len(tab)
    if n_missing > 0:
        # Create an empty table with same columns
        extra_rows = Table({col: np.full(n_missing, fill_value, dtype=tab[col].dtype)
                            for col in tab.colnames})
        tab = vstack([tab, extra_rows])
    return tab


def load_lae_candidates():
    ml_cols = ['massformed',
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
                'EW_lya_rest']

    ceers_laes = Table.read('cand_laes/Final_LAEs_CEERS.fits')['New_IDs'].value
    goodsn_laes = Table.read('cand_laes/Final_LAEs_GOODSN.fits')['New_IDs'].value
    shela_laes = Table.read('cand_laes/Final_LAEs_SHELA.fits')['New_IDs'].value
    tone_laes = Table.read('cand_laes/Final_LAEs_TONE_v2.fits')['New_IDs'].value

    #fields = ['CEERS', 'GOODSN', 'SHELA', 'TONE']
    fields = ['GOODSN', 'SHELA', 'CEERS', 'TONE']
    tabs = []
    
    #for field, laes in zip(fields, [ceers_laes, goodsn_laes, shela_laes, tone_laes]):
    for field, laes in zip(fields, [goodsn_laes, shela_laes, ceers_laes, tone_laes]):
        for ids in tqdm(laes, total = len(laes), desc=f'Loading LAEs for {field}'):
            tab = load_posterior_distribution(field, ids)
            tab = pad_table(tab, target_rows = 500, fill_value=0)
            df = tab.to_pandas()
            tabs.append(df[ml_cols].values)

    
    #merged_ids = np.concatenate((ceers_laes, goodsn_laes, shela_laes, tone_laes))
    merged_ids = np.concatenate((goodsn_laes, shela_laes, ceers_laes, tone_laes))

    return merged_ids, np.array(tabs)


def grab_random_rows(tables):

    n_galaxies, n_rows, n_features = tables.shape
    rand_idx = np.random.randint(0, n_rows, size=n_galaxies)
    random_rows_per_galaxy = tables[np.arange(n_galaxies), rand_idx, :]

    return random_rows_per_galaxy

def test():
     
    ml_cols = ['massformed',
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
                'EW_lya_rest']

    ceers_laes = np.loadtxt('cand_laes/CEERS_LAE_IDs.txt').astype(int)
    goodsn_laes = np.loadtxt('cand_laes/GOODSN_LAE_IDs.txt').astype(int)

    fields = ['CEERS', 'GOODSN']
    tabs = []
    
    for field, laes in zip(fields, [ceers_laes, goodsn_laes]):
        #print(f'Loading {field} LAEs')
        for ids in tqdm(laes, total = len(laes), desc=f'Loading LAEs for {field}'):
            tab = load_posterior_distribution(field, ids)
            tab = pad_table(tab, target_rows = 500, fill_value=0)
            df = tab.to_pandas()
            tabs.append(df[ml_cols].values)

    merged_ids = np.concatenate((ceers_laes, goodsn_laes))

    return merged_ids, np.array(tabs)

    


with open('bnn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def grab_features_and_predictor_from_config():

    """
    Extracts features and predictor from the configuration file.
    Returns
    -------
    tuple
        A tuple containing the features DataFrame and the predictor Series.
    """

    input_data = config['input_data']
    features = input_data['ml_features']
    predictor = input_data['ml_predictor']

    return features, predictor

features, predictor = grab_features_and_predictor_from_config()

def read_data(file_path):

    return Table.read(file_path)

def grab_features_and_predictor(data_table, log_y=False):

    """
    Extracts features and predictor from the data table.
    Parameters
    ----------
    data_table : astropy.table.Table
        The table containing the data.
    Returns
    -------
    tuple    

        A tuple containing the features DataFrame and the predictor Series.
    """
    features_df = data_table[features].to_pandas()
    predictor_series = data_table[predictor].to_pandas().squeeze()

    features_df['gal_ID'] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in features_df['gal_ID']]

    if log_y:
        predictor_series = np.log10(predictor_series)

    return features_df, predictor_series


def split_data(features_df, predictor_series, how = 'galaxy', test_size=0.2, random_state=42):

    if how == 'random':
        X_train, X_test, y_train, y_test = train_test_split(features_df,
                                                            predictor_series,
                                                            test_size=test_size,
                                                            random_state=random_state)
    elif how == 'galaxy':
        unique_galaxies = features_df['survey'].unique()
        train_galaxies, test_galaxies = train_test_split(unique_galaxies,
                                                         test_size=test_size,
                                                         random_state=random_state)
        X_train = features_df[features_df['survey'].isin(train_galaxies)]
        X_test = features_df[features_df['survey'].isin(test_galaxies)]
        y_train = predictor_series[X_train.index]
        y_test = predictor_series[X_test.index] 

    else:
        raise ValueError("Invalid 'how' parameter. Use 'random' or 'galaxy'.")

    return X_train, X_test, y_train, y_test

def scale_data(X):

    """
    Scales the features using StandardScaler.
    Parameters
    ----------
    X : pandas.DataFrame
        The features DataFrame to be scaled.
    Returns
    -------
    pandas.DataFrame
        The scaled features DataFrame.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)

    return data_scaled

def drop_columns(df):
    cols = df.columns
    if 'gal_ID' in cols:
        df = df.drop(columns=['gal_ID'])
    if 'chisq_phot' in cols:
        df = df.drop(columns=['chisq_phot'])
    if 'perturbed_flux' in cols:
        df = df.drop(columns=['perturbed_flux'])
    if 'lya_continuum' in cols:
        df = df.drop(columns=['lya_continuum'])

    return df

def scale_training_data(X_train, y_train = None):

    cols = X_train.columns
    if 'gal_ID' in cols:
        X_train = X_train.drop(columns=['gal_ID'])
    if 'chisq_phot' in cols:
        X_train = X_train.drop(columns=['chisq_phot'])
    if 'perturbed_flux' in cols:
        X_train = X_train.drop(columns=['perturbed_flux'])
    if 'lya_continuum' in cols:
        X_train = X_train.drop(columns=['lya_continuum'])
    
    cols = X_train.columns
    scaled_Xtrain = scale_data(X_train)

    if y_train is not None:
        scaled_ytrain = scale_data(y_train.values.reshape(-1, 1)).flatten()
        return scaled_Xtrain, scaled_ytrain
    
    return scaled_Xtrain, cols

def load_data():

    files = config['input_data']['files']

    data_tables = [read_data(file) for file in files]

    stacked_table = vstack(data_tables)

    survey = [x.decode('utf-8').split('_')[0] for x in stacked_table['gal_ID'].data]

    stacked_table['survey'] = survey

    return stacked_table

