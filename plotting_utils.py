import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import itertools
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
import seaborn as sb
import yaml 

with open('plot_config.yaml', 'r') as f:
    plot_config = yaml.safe_load(f)

plot_params = plot_config['plot_params']

plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

DPI = 150


map_text = {'continuity:dsfr1': r'$\Delta$SFR$_1$ [M$_\odot$ yr$^{-1}$]',
            'continuity:dsfr2': r'$\Delta$SFR$_2$ [M$_\odot$ yr$^{-1}$]',
            'continuity:dsfr3': r'$\Delta$SFR$_3$ [M$_\odot$ yr$^{-1}$]',
            'continuity:dsfr4': r'$\Delta$SFR$_4$ [M$_\odot$ yr$^{-1}$]',   
            'continuity:dsfr5': r'$\Delta$SFR$_5$ [M$_\odot$ yr$^{-1}$]',
            'continuity:dsfr6': r'$\Delta$SFR$_6$ [M$_\odot$ yr$^{-1}$]',
            'continuity:dsfr7': r'$\Delta$SFR$_7$ [M$_\odot$ yr$^{-1}$]',
            'stellar_mass': r'M$_*$ [M$_\odot$]',
            'formed_mass': r'M$_{formed}$ [M$_\odot$]',
            'mass_weighted_age': r'Age$_{mw}$ [Gyr]',
            'tform': r'T$_{form}$ [Gyr]',
            'tquench': r'T$_{quench}$ [Gyr]',   
            'continuity:massformed': r'M$_\odot$',
            'continuity:metallicity': r'Z/Z$_\odot$',
            'sfr': r'SFR [M$_\odot$ yr$^{-1}$]',
            'mass': r'M$_*$ [M$_\odot$]',
            'nebular:logU': r'log$_{10}$(U)',
            'dust:Av': r'A$_V$ [mag]',
            'dust:delta': r'$\Delta$ [mag]',
            'dust:B': r'B',
            'ssfr': r'sSFR [yr$^{-1}$]',
            'Muv': r'M$_{UV}$ [mag]',
            'beta': r'$\beta$',
            'burstiness': r'log$_{10}$(SFR$_{10}$/SFR$_{100}$)', 
            'mass_weighted_zmet': r'Z$_{mw}$ [Z/Z$_\odot$]', 
            'chisq_phot': r'$\chi^2_{phot}$', 
            'EW_lya_rest': r'EW$_{Ly\alpha}$ [\AA]'}

def plot_sfh(ages, sfh, title='Star Formation History', xlabel='Age [Myr]', ylabel=r'SFR [M$_{\odot} yr^{-1}$]', save_path=None):

    """    Plots the Star Formation History (SFH) given ages and SFR values.
    Args:
        ages (list or array): Ages in Myr.
        sfh (list or array): Star Formation Rate (SFR) values in M_sun/yr.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plot_ages = ages / 1e6  # Convert ages from years to Myr
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(sfh.shape) == 2:
        # If sfh is 2D, plot l16, med, and u84

        l16, med, u84 = np.percentile(sfh, [16, 50, 84], axis=0)
        ax.fill_between(plot_ages, l16, u84, color='gray', alpha=0.5, label='16-84% range', step = 'mid')
        ax.step(plot_ages, med, where = 'mid', color='black', label='Median SFR')
    else:
        # If sfh is 1D, plot directly
        ax.plot(plot_ages, sfh, color='black', label='SFR')

    ax.legend()
    ax.grid(True)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    else:
        plt.tight_layout()
        return fig, ax


def plot_bagpipes_models(sed_wave, sed, 
                         photom_wave, photom_flux, chi2, redshift,
                         data_phot_flux = None, data_phot_err = None, 
                         snr_threshold = 1,
                         kind = 'median',
                         title='SED', 
                         xlabel='Wavelength (Angstrom)',
                         ylabel='Flux Density (erg/s/cm^2/Angstrom)', 
                         save_path=None):

    """    Plots the SED models given wavelength and flux density values.
    Args:
        sed_wave (list or array): Wavelength values in Angstrom.
        sed (list or array): Flux density values in erg/s/cm^2/Angstrom
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): If provided, saves the plot to this path.
    """     
    min_chi2 = np.amin(chi2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(rf'{title} @ z={redshift:.2f}, $\chi^2_{{min}}$: {min_chi2:.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ###############################
    #Plotting the SED
    ###############################
    if len(sed.shape) == 2:
        if kind == 'median':
            # If sed is 2D, plot l16, med, and u84
            l16, med, u84 = np.percentile(sed, [16, 50, 84], axis=0)
            ax.fill_between(sed_wave, l16, u84, color='gray', alpha=0.5, label='16-84% range')
            ax.plot(sed_wave, med, color='black', label='Median SED')
        elif kind == 'min_chi2':
            # If sed is 2D, plot the model with the minimum chi2
            min_chi2_index = np.argmin(chi2)
            min_chi2 = chi2[min_chi2_index]
            ax.plot(sed_wave, sed[min_chi2_index], color='black', label=rf'$\chi^2_{{min}}$: {min_chi2:.2f}')

        else:
            raise ValueError("Invalid kind specified. Use 'Median' or 'min_chi2'.")
    else:
        # If sed is 1D, plot directly
        min_chi2_index = np.argmin(chi2)
        min_chi2 = chi2[min_chi2_index]
        ax.plot(sed_wave, sed, color='black',  label=rf'$\chi^2_{{min}}$: {min_chi2:.2f}')

    ###############################
    #Plotting the Photometry
    ###############################
    if len(photom_flux.shape) == 2:
        # If photom_flux is 2D, plot l16, med, and u84
        l16, med, u84 = np.percentile(photom_flux, [16, 50, 84], axis=0)

        width_fraction = 0.05  # Fraction of the wavelength to use for the fill width
        for i, lam in enumerate(photom_wave):
            
            delta = lam * width_fraction
            x_fill = np.array([lam - delta, lam + delta])
            y_low = np.array([l16[i], l16[i]])
            y_high = np.array([u84[i], u84[i]])
    
            ax.fill_between(x_fill, y_low, y_high, color='skyblue', alpha=0.6)

        ax.scatter(photom_wave , med, color='purple', label = 'Model Photometry')

    else:
        # If photom_flux is 1D, plot directly
        ax.scatter(photom_wave, photom_flux, color='purple',  label = 'Model Photometry') 

    if data_phot_err is not None:
        #print('Real Data Photometry Detected')
        snr = data_phot_flux / data_phot_err
        is_upper_limit = snr < snr_threshold

        phot_flux_mujy_to_fnu = data_phot_flux * u.microJansky.to(u.erg / u.s / u.cm**2 / u.Hz)  # Convert to F_nu
        phot_fluxerr_mujy_to_fnu = data_phot_err * u.microJansky.to(u.erg / u.s / u.cm**2 / u.Hz) 

        phot_flux_flambda = phot_flux_mujy_to_fnu * ((3e18 * u.Angstrom/u.s) / (photom_wave*u.Angstrom)**2).value  # Convert to F_nu
        phot_fluxerr_flambda = phot_fluxerr_mujy_to_fnu * ((3e18 * u.Angstrom/u.s) / (photom_wave*u.Angstrom)**2).value # Convert to F_nu error

        # Plot downward arrows for upper limits
        for i, is_ul in enumerate(is_upper_limit):
            if is_ul:
                flux_ul = phot_fluxerr_flambda[i]  # We will plot just the 1 sigma error for upper limits
                ax.errorbar(photom_wave[i], flux_ul,
                            yerr=flux_ul * 0.2,  # short arrow size
                            uplims = True, lolims=False, color='black', fmt='o', markersize=5)
            else:
                ax.errorbar(photom_wave[i], phot_flux_flambda[i],
                            yerr=phot_fluxerr_flambda[i], fmt='o', color='black', markersize=5, capsize=3)

    xmin = photom_wave.min() * 0.8
    xmax = photom_wave.max() * 1.2

    good_flux = phot_flux_flambda[phot_flux_flambda > 0]

    #checking the length of good_flux to avoid errors
    if len(good_flux) == 0:
        good_flux = phot_flux_flambda


    min_y_data = np.amin(good_flux)
    min_y_model = np.amin(photom_flux)

    max_y_data = np.amax(good_flux)
    max_y_model = np.amax(photom_flux)


    ymin = min(min_y_data, min_y_model) * 0.8
    ymax = max(max_y_data, max_y_model) * 1.5

    ax.set_ylim([ymin, ymax])  # Set y-axis limits
    ax.set_xlim([xmin, xmax])  # Set x-axis limits


    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi = DPI)

    else:
        plt.tight_layout()
        return fig, ax

def plot_summary_table(tab):

    columns = tab.colnames[1:]
    unique_columns = ['_'.join(x.split('_')[1:]) for x in columns ]
    sorted_cols = np.sort(np.unique(unique_columns))

    lerr = []
    uerr = []

    for col in sorted_cols:
        lerror = tab[f'l16_{col[0]}'].data[0]
        med = tab[f'med_{col[0]}'].data[0]
        uerror = tab[f'u84_{col[0]}'].data[0]

        lerr.append(np.abs(med - lerror))
        uerr.append(np.abs(uerror - med))

    fig, axes = plt.subplots(nrows = 5, ncols = 5, figsize=(10, 5))

    axes = axes.flatten()  # Flatten the 2D array of axes to 1D

    for i, col in enumerate(sorted_cols):
        ax = axes[i]
        ax.errorbar(med, i, xerr=[[lerr[i]], [uerr[i]]], fmt='o', color='blue', label=col)
        ax.set_yticks([])
        ax.set_xlabel(map_text.get(col, col))  # Use map_text for labels
        ax.set_title(col)


def multipanel_histogram(tab, nrows, ncols, bins=30, titles=None, ylabel='Frequency', save_path=None):

    """    Plots a multipanel histogram for the given fit object.
    Args:
        fit (object): Fit object containing the data to plot.
        nrows (int): Number of rows in the multipanel plot.          
        ncols (int): Number of columns in the multipanel plot.
        bins (int): Number of bins for the histogram.
        titles (list of str, optional): Titles for each subplot. If None, uses default
            titles based on the fit object.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): If provided, saves the plot to this path.
    """

    #check to make sure the columns of the tab match or are less than nrows * ncols
    if len(tab.columns) > nrows * ncols:
        raise ValueError("Number of columns in the table exceeds nrows * ncols. Please adjust the parameters.")
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes to 1
    columns = tab.colnames

    for i, col in enumerate(tab.columns):
        
        ax = axs[i]
        col = columns[i]
        data = tab[col]
        ax.hist(data, bins=bins, color='blue', alpha=0.7)
        ax.set_xlabel(map_text.get(col, col))  # Use map_text for labels if available
        ax.set_ylabel(ylabel)
        ax.grid(True)

    for j in range(len(columns), len(axs)):
        fig.delaxes(axs[j])
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi = DPI)
    
    else:

        plt.tight_layout()
        return fig, axs



def plot_histogram_data(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency', save_path=None):
    """    Plots a histogram of the given data.
    Args:
        data (list or array): Data to plot.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): If provided, saves the plot to this path.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=bins, color='purple', alpha=0.7)
    #ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi = DPI)
    else:
        plt.tight_layout()
        return fig, ax



def corner_plot(tab, labels=None, truths=None, bins=20, figsize=(15, 15), hist_kwargs=None):
    """
    Create a corner plot using matplotlib.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_dim)
        The MCMC samples.
    labels : list of str
        List of parameter names for axis labels.
    truths : list of float
        List of "true" values to overlay.
    bins : int
        Number of bins in histograms.
    figsize : tuple
        Size of the figure.
    hist_kwargs : dict
        Additional kwargs for histogram plotting.
    """
    samples = np.vstack([tab[name] for name in tab.colnames]).T
    labels = tab.colnames
    n_dim = samples.shape[1]
    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize)
    hist_kwargs = hist_kwargs or {}

    for i, j in itertools.product(range(n_dim), repeat=2):
        ax = axes[i, j]
        x = samples[:, j]
        y = samples[:, i]

        if i == j:
            # Diagonal: 1D histogram
            counts, bins_hist = np.histogram(x, bins=bins, density=True)
            ax.hist(x, bins=bins, density=True, color='purple', alpha=0.6, **hist_kwargs)
            ax.grid(True)
            #sb.histplot(x, bins=bins, kde=True, ax=ax, color='k', alpha=0.6, stat='density', **hist_kwargs)
            ax.set_ylim(0, 1.1 * counts.max())

            low_x = x.min() - 0.5*np.std(x)
            high_x = x.max() + 0.5*np.std(x)

            #assert low_x < high_x, "Low x value must be less than high x value."
            if low_x < high_x:
                ax.set_xlim(low_x, high_x)
            else:
                ax.set_xlim(x.min() - 0.5*np.std(x), x.max() + 0.5*np.std(x))
            ax.set_yticks([])
            ax.set_yticklabels([])
            if truths is not None:
                ax.axvline(truths[j], color='red', lw=1.5, ls='--')

        elif i > j:
            # Lower triangle: 2D KDE plot
            try:
                #good_data = np.isfinite(x) & np.isfinite(y)
                x = x#[good_data]
                y = y#[good_data]
                kde = gaussian_kde([x, y])
                xx, yy = np.meshgrid(
                    np.linspace(x.min(), x.max(), 100),
                    np.linspace(y.min(), y.max(), 100)
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                zz_flat = zz.flatten()
                zz_sorted = np.sort(zz_flat)
                cdf = np.cumsum(zz_sorted)
                cdf /= cdf[-1]

                percentiles = [0.16, 0.50, 0.84]
                levels = [zz_sorted[np.searchsorted(cdf, p)] for p in percentiles]
                ax.scatter(x, y, marker = 'o', s = 5, alpha=0.5, color = 'gray')
                ax.contourf(xx, yy, zz, levels=levels, cmap='plasma', alpha = 0.5)
                
            except np.linalg.LinAlgError:
                ax.scatter(x, y, marker = 'o', s = 5, alpha=0.5)

            ax.set_xlim(x.min(), x.max())

            ymin, ymax = y.min(), y.max()
            if np.isclose(ymin, ymax):
                # Expand manually to avoid singularity
                delta = 1e-3
                ax.set_ylim(ymin - delta, ymax + delta)
            else:
                ax.set_ylim(ymin, ymax)

            if truths is not None:
                ax.plot(truths[j], truths[i], 'rx')

        else:
            ax.axis('off')

        # Tick labels
        if i < n_dim - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(map_text.get(labels[j], labels[j]), fontsize = 7)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

        if j > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(map_text.get(labels[i], labels[i]), fontsize = 7)
            for label in ax.get_yticklabels():
                label.set_rotation(45)
                label.set_va('top')

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    
    return fig

def plot_histogram(table, column, ax = None, save_path=None):
    """    Plots a histogram for a specific column in the table.    

    """

    df = table.to_pandas()
    minx = plot_params[column]['min']
    maxx = plot_params[column]['max']
    bins = plot_params[column]['bins']
    
    if ax is None:

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Histogram of {column}')
        df[column].hist(bins=bins, range=(minx, maxx), color='purple', alpha=0.7)
        ax.set_xlabel(map_text.get(column, column))
        ax.set_ylabel('Frequency')

        if save_path:
            
            plt.savefig(save_path, dpi=DPI)
        
        else:

            plt.tight_layout()
            return fig, ax

    else:

        ax.set_title(f'Histogram of {column}')
        df[column].hist(bins=bins, range=(minx, maxx), color='purple', alpha=0.7, ax=ax)
        ax.set_xlabel(map_text.get(column, column))
        ax.set_ylabel('Frequency')


def plot_filters(filter_paths, field):

    lowz = 1.9
    highz = 3.5
    lya_rest  = 1215.67  # Lyman-alpha rest wavelength in Angstroms

    low_lya = lya_rest * (1 + lowz)  # Lyman-alpha wavelength at high redshift
    med_lya = lya_rest * (1 + 2.5)  # Lyman-alpha wavelength at medium redshift
    high_lya = lya_rest * (1 + highz)  # Lyman-alpha wavelength at high redshift
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(low_lya, color='red', linestyle='--', )
    ax.axvline(med_lya, color='orange', linestyle='--', )
    ax.axvline(high_lya, color='blue', linestyle='--',)
    for filt in filter_paths:
        
        wave, tlam = np.loadtxt(filt, unpack = True)
        
        good_data = tlam > 1e-3

        wave = wave[good_data]
        tlam = tlam[good_data]

        mask = (wave[0] < low_lya) & (low_lya < wave[-1]) | (wave[0] < high_lya) & (high_lya < wave[-1]) | (wave[0] < med_lya) & (med_lya < wave[-1])

        if mask:
            ax.fill_between(wave, tlam, alpha=0.2, label = filt.split('/')[-1])
        else:
            ax.plot(wave, tlam)
    
    ax.set_xlabel('Wavelength (Angstrom)')
    ax.set_ylabel('Transmission')
    ax.set_ylim(0, 1)
    ax.legend(ncols = 2)
    fig.savefig(f'cand_lae_plots/{field}_filters.png', dpi=150)
    plt.close('all')
