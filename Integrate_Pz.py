from astropy.table import Table, join, vstack, hstack
import numpy as np
import pandas as pd
from astropy.io import fits
import argparse
from scipy.integrate import trapezoid
from tqdm import tqdm
field = 'CEERS'
ZMIN = 0
ZMAX = 15
DELTAZ = 0.2

zarr = pd.read_csv('z_arr_TONE.csv')['z_arr'].values

def load_fits_file(file):
    """
    Load a FITS file and return the HDUList.
    
    Parameters
    ----------
    file : str
        Path to the FITS file.
    
    Returns
    -------
    astropy.io.fits.HDUList
        The HDUList of the FITS file.
    """
    try:
        hdu = fits.open(file)
        return hdu
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        raise

def load_hdr_id(field):

    """
    Load the header ID table for a given field.
    
    Parameters
    ----------
    field : str
        The field name to load the header ID table for.
    
    Returns
    -------
    astropy.table.Table
        The header ID table.
    """
    try:
        hdr_tab = Table.read(f'Matched_Catalogs/{field}_matched_hdr_df.fits')
        return hdr_tab
    except Exception as e:
        print(f"Error loading header ID table: {e}")
        raise


def grab_summary(hdu):

    """
    Extract the summary table from the FITS HDUList.
    
    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        The FITS HDUList containing the summary data.
    
    Returns
    -------
    astropy.table.Table
        The summary table.
    """
    try:
        summary_tab = Table(hdu['SUMMARY'].data)
        return summary_tab
    except KeyError:
        print("No 'SUMMARY' extension found in the FITS file.")
        raise

def grab_pz(hdu, field):
    """
    Extract the P(z) data from the FITS HDUList.
    
    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        The FITS HDUList containing the P(z) data.
    
    Returns
    -------
    tuple
        A tuple containing the redshift array and a table with P(z) values.
    """
    try:
        
        pz_arr = Table(hdu['PZ'].data)['Pz'][1:].data
        return pz_arr
    
    except KeyError:
        print("No 'PZ' extension found in the FITS file.")
        raise

def grab_ID(hdu):
    """
    Extract the ID from the FITS HDUList.
    
    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        The FITS HDUList containing the ID data.
    
    Returns
    -------
    astropy.table.Table
        A table with the ID data.
    """
    try:
        summary = Table(hdu['SUMMARY'].data)
        IDs = summary['ID'].astype(int)
        return IDs
    except KeyError:
        print("No 'ID' extension found in the FITS file.")
        raise

def make_hdr_pz(tab, z_arr):

    plya = tab['plya_classification']

    low_z = (z_arr < 1.9)
    high_z = (z_arr >= 1.9)

    new_pz_list = []

    for i in range(len(tab)):
        
        new_pz = np.zeros(len(z_arr))
        new_pz[low_z] = 1 - plya[i]
        new_pz[high_z] = plya[i]

        normalized_new_pz = new_pz / np.sum(new_pz) 

        new_pz_list.append(normalized_new_pz)

    tab['new_pz'] = new_pz_list

    return tab

def integrate_pz_in_bins(z_arr, pz_arr, ID, bin_edges = np.arange(ZMIN, ZMAX+DELTAZ, DELTAZ)):
    """
    Integrate P(z) over specified redshift bins.

    Parameters
    ----------
    z_arr : 1D np.ndarray
        Redshift grid (must be same length as pz_arr).
    pz_arr : 1D np.ndarray
        P(z) values (same length as z_arr).
    bin_edges : array-like
        Edges of the redshift bins (e.g., [0.0, 0.2, 0.4, ..., 1.0]).

    Returns
    -------
    integrals : list of float
        Integrated probability in each redshift bin.
    """
    z_arr = np.asarray(z_arr)
    pz_arr = np.asarray(pz_arr)
    bin_edges = np.asarray(bin_edges)

    integrals = []
    
    oii_mask = z_arr < 0.5
    lya_mask = (z_arr >= 1.5) & (z_arr < 4.0)
    
    oii_integral = trapezoid(y = pz_arr[oii_mask], x = z_arr[oii_mask])
    lya_integral = trapezoid(y = pz_arr[lya_mask], x = z_arr[lya_mask])
    
    integrals.append(oii_integral)
    integrals.append(lya_integral)

    for i in range(len(bin_edges) - 1):
        zmin, zmax = bin_edges[i], bin_edges[i + 1]
        # Mask points within the bin
        mask = (z_arr >= zmin) & (z_arr < zmax)
        if np.any(mask):
            integral = trapezoid(y = pz_arr[mask], x = z_arr[mask])
        else:
            integral = 0.0
        integrals.append(integral)


    bin_int = ['OII Integral', 'Lya Integral']

    bin_label = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

    bin_labels = bin_int + bin_label
    tab_dict = {x:[y] for x, y in zip(bin_labels, integrals)}
    
    max_integral = np.argmax(integrals[2:]) + 2  # Skip OII and Lya integrals
    tab_dict['Max Integral Bin'] = [bin_labels[max_integral]]
    tab_dict['Max Integral Value'] = [integrals[max_integral]]
    
    table = Table(tab_dict)

    return table


def grab_pz_integrated(hdu, bin_edges=np.arange(ZMIN, ZMAX+DELTAZ, DELTAZ)):
    """
    Extract and integrate the redshift probability distribution from a FITS HDU.

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        The FITS HDU containing the P(z) data.
    bin_edges : array-like, optional
        Edges of the redshift bins for integration (default is np.arange(0, 6.2, .2)).

    Returns
    -------
    Table
        A table with integrated P(z) values over specified bins.
    """
    pz_arr = grab_pz(hdu, field)
    IDs = grab_ID(hdu)

    tabs = []

    for ids, pz in tqdm(zip(IDs, pz_arr), total=len(IDs), desc="Integrating p(z)"):
        tab = integrate_pz_in_bins(zarr, pz, ids, bin_edges)
        tabs.append(tab)
    integrated_table = vstack(tabs)
    return integrated_table

def main_integrals(file):

    """
    Main function to extract and integrate P(z) from a FITS file.   
    Parameters
    ----------
    file : str

        Path to the FITS file containing P(z) data.
    Returns
    -------
    Table
        A table with integrated P(z) values over specified bins.
    """
    hdu = load_fits_file(file)
    return grab_pz_integrated(hdu)


def main_pz_merge(file, save_path=None):

    hdu = load_fits_file(file)
    summary_tab = grab_summary(hdu)
    pz_tab = grab_pz(hdu, field)
    integrals = grab_pz_integrated(hdu)
    IDs = grab_ID(hdu).data

    columns = ['ID', 'z_best', 'chi2', 'z_l95', 'z_l68', 'z_med', 'z_u68', 'z_u95']

    merged_pz_summary_tab = hstack((summary_tab[columns], integrals))

    lya_gt_OII = merged_pz_summary_tab['Lya Integral'] > merged_pz_summary_tab['OII Integral']
    merged_pz_summary_tab['Lya_gt_OII'] = lya_gt_OII
    lya_gt_0pt7 = merged_pz_summary_tab['Lya Integral'] > 0.7
    merged_pz_summary_tab['Lya_gt_0pt7'] = lya_gt_0pt7

    if save_path:
        merged_pz_summary_tab.write(save_path, overwrite=True)
        print(f"Saved merged table to {save_path}")
    else:
        print("No save path provided, returning merged table without saving.")
        return merged_pz_summary_tab


def main_pz_merge_new(file, save_path=None):

    hdu = load_fits_file(file)
    summary_tab = grab_summary(hdu)
    pz_tab = grab_pz(hdu, field)
    integrals = grab_pz_integrated(hdu)
    IDs = grab_ID(hdu).data

    columns = ['ID', 'z_best', 'chi2', 'z_l95', 'z_l68', 'z_med', 'z_u68', 'z_u95']

    merged_pz_summary_tab = hstack((summary_tab[columns], integrals))

    lya_gt_OII = merged_pz_summary_tab['Lya Integral'] > merged_pz_summary_tab['OII Integral']
    merged_pz_summary_tab['Lya_gt_OII'] = lya_gt_OII
    lya_gt_0pt7 = merged_pz_summary_tab['Lya Integral'] > 0.7
    merged_pz_summary_tab['Lya_gt_0pt7'] = lya_gt_0pt7

    if save_path:
        merged_pz_summary_tab.write(save_path, overwrite=True)
        print(f"Saved merged table to {save_path}")
    else:
        print("No save path provided, returning merged table without saving.")
        return merged_pz_summary_tab


def set_up_pz_proper(hdr_file, pz_file):

    hdr_tab = Table.read(hdr_file)
    pz_tab = Table(fits.open(pz_file)['PZ'].data)
    hdu = fits.open(pz_file)
    IDs = grab_ID(hdu).data

    pz_tab = pz_tab[1:]

    hdr_pz_tab = make_hdr_pz(hdr_tab, zarr)

    pz_tab['new_pz'] = hdr_pz_tab['new_pz']
    pz_tab['norm_pz'] = [x/x.sum() for x in pz_tab['Pz']]
    pz_tab['merged_pz'] = [(x*y)/np.sum(x*y) for x, y in zip(pz_tab['norm_pz'], pz_tab['new_pz'])]
    summary_tab = grab_summary(hdu)

    return pz_tab, IDs, summary_tab

def integrate_pz_tab(pz_tab, IDs, summary_tab):

    tabs = []
    
    for pz, ids in tqdm(zip(pz_tab['merged_pz'], IDs), total=len(IDs), desc="Integrating P(z)"):
        tab = integrate_pz_in_bins(zarr, pz, ids)
        tabs.append(tab)
    
    integrated_table = vstack(tabs)

    columns = ['ID', 'z_best', 'chi2', 'z_l95', 'z_l68', 'z_med', 'z_u68', 'z_u95']


    merged_pz_summary_tab = hstack((summary_tab[columns], integrated_table))

    lya_gt_OII = merged_pz_summary_tab['Lya Integral'] > merged_pz_summary_tab['OII Integral']
    merged_pz_summary_tab['Lya_gt_OII'] = lya_gt_OII
    lya_gt_0pt7 = merged_pz_summary_tab['Lya Integral'] > 0.007
    merged_pz_summary_tab['Lya_gt_0pt7'] = lya_gt_0pt7

    return merged_pz_summary_tab



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate P(z) from a FITS file.")
    parser.add_argument("--field", type=str, help="Field of the FITS file.")
    parser.add_argument("--save_path", type=str, help="Path to save the merged table (optional).")
    
    args = parser.parse_args()
    
    if args.save_path:
        main_pz_merge(args.file, args.save_path)
    else:
        result = main_integrals(args.file)
        print(result)
