"""
Galactic Astronomy, HW3
Due: February 28
Jonas Powell
"""

import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from sklearn.linear_model import LinearRegression
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia


def get_data(metric='distance', n_sources=1000):
    if metric not in ['distance', 'brightness']:
        return "Please choose 'distance' or 'brightness'."

    # Translate english to SQL/Gaia and call.
    param = 'parallax' if metric is 'distance' else 'lum_val'
    if metric is 'distance':
        # Want to remove sources closer than Alpha Cen, whose parallax angle is 0.768"
        gaia_str = "SELECT top {} * FROM gaiadr2.gaia_source \
                    WHERE parallax > 700 \
                    AND phot_rp_mean_mag != 'Nan' \
                    AND bp_rp != 'Nan' \
                    AND lum_val != 'Nan' \
                    ORDER BY parallax DESC \
                    ".format(n_sources)
    else:
        gaia_str = "SELECT top {} * FROM gaiadr2.gaia_source \
                    WHERE 'lum_val' > 0 \
                    AND phot_rp_mean_mag != 'Nan' \
                    AND bp_rp != 'Nan' \
                    AND lum_val != 'Nan' \
                    ORDER BY lum_val DESC \
                    ".format(n_sources)


    job = Gaia.launch_job(gaia_str)  #, dump_to_file=True)
    gaia_results_raw = job.get_results()

    gaia_results = gaia_results_raw.to_pandas()

    sort_feature = 'Distance' if metric is 'distance' else 'Absolute Magnitude'
    df = pd.DataFrame()
    df['Distance'] = (gaia_results['parallax'] * 1e-3)**(-1)
    df['mag'] = gaia_results['phot_rp_mean_mag']
    df['Color'] = gaia_results['bp_rp']
    df['Absolute Magnitude'] = df['mag'] - \
        5 * (np.log10(df['Distance']) - 1)
    df['T Effective'] = gaia_results['teff_val']
    df['Parallax'] = gaia_results['parallax']
    df['Radius'] = gaia_results['radius_val']
    df['Luminosity'] = gaia_results['lum_val']
    df['Mass'] = df['Luminosity']**0.25
    df['Sort Feature'] = sort_feature
    df['Decimal Sort Feature'] = df[sort_feature]/np.nanmax(df[sort_feature])

    df.to_csv('stars_by_{}-{}.csv'.format(metric, n_sources))
    return df




# Problem 3
def make_plots(n_sources=1000, save=False):
    """Make the necessary plots for each dataset."""
    plt.clf()
    cmap1, cmap2 = 'Reds', 'Blues'

    # If we haven't already downloaded the data, get it.
    try:
        closest = pd.read_csv('stars_by_distance-{}.csv'.format(n_sources))
        print "Already have distance data; reading it in now."
    except IOError:
        closest = get_data(metric='distance', n_sources=n_sources)
        print "Don't yet have distance data yet; downloading now."

    try:
        brightest = pd.read_csv('stars_by_brightness-{}.csv'.format(n_sources))
        print "Already have brightness data; reading it in now."
    except IOError:
        brightest = get_data(metric='brightness', n_sources=n_sources)
        print "Don't yet have brightness data yet; downloading now."

    df = pd.concat([closest, brightest])

    # Get the plots rolling.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Shade by density
    ax = sns.kdeplot(closest['Color'], closest['Absolute Magnitude'],
                     cmap=cmap1, shade=True, shade_lowest=False)
    ax = sns.kdeplot(brightest['Color'], brightest['Absolute Magnitude'],
                     cmap=cmap2, shade=True, shade_lowest=False)

    # Plot contours
    # ax = sns.kdeplot(closest['Color'], closest['Absolute Magnitude'], n_levels=30, alpha=0.5, cmap=cmap1)
    # ax = sns.kdeplot(brightest['Color'], brightest['Absolute Magnitude'], n_levels=30, alpha=0.5, cmap=cmap2)

    # Scatter on the points
    ax = sns.scatterplot('Color', 'Absolute Magnitude', data=closest, ax=ax,
                         alpha=0.2, color='darkred', linewidth=0, label='Closest Stars')
    ax = sns.scatterplot('Color', 'Absolute Magnitude', data=brightest, ax=ax,
                         alpha=0.1, color='darkblue', linewidth=0, label='Brightest Stars')


    ax.set_xticks(np.linspace(min(df['Color']), max(df['Color']), 4))
    ax.set_yticks(np.linspace(min(df['Absolute Magnitude']),
                              max(df['Absolute Magnitude']), 4))

    ax.set_ylim(ax.get_xlim()[::-1])
    ax.set_title('HR Diagram', weight='bold')
    plt.legend(frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.1)
    if save is True:
        outname = 'HR_diagram.pdf'
        plt.savefig(outname, dpi=200)
        print "Saved to", outname
    else:
        print "Showing:"
        plt.show(block=False)



# Problem 4
n_sources=1000
def get_densities(n_sources=1000):
    # If we haven't already downloaded the data, get it.
    try:
        df = pd.read_csv('stars_by_distance-{}.csv'.format(n_sources))
        print "Already have distance data; reading it in now."
    except IOError:
        df = get_data(metric='distance', n_sources=n_sources)
        print "Don't yet have distance data yet; downloading now."

    msol_to_gm = 1.989e33
    pc_to_cm = 3.08e18
    gm_to_nH = 1/(1.674e-24)
    n_stars = len(df['Distance'])

    n_per_pc3 = n_stars / np.nanmax(df['Distance'])**3
    m_sol_per_pc3 = np.sum(df['Mass']) / np.nanmax(df['Distance'])**3
    gm_per_cm3 = msol_to_gm * np.sum(df['Mass']) / (pc_to_cm * np.nanmax(df['Distance']))**3

    nH = np.sum(n_stars * df['Mass']) * msol_to_gm * gm_to_nH
    nH_per_cm3 = nH / (pc_to_cm * np.nanmax(df['Distance']))**3

    print "Stars per Cubic Parsec: ", n_per_pc3
    print "Solar Masses per Cubic Parsec: ", m_sol_per_pc3
    print "Grams per Cubic Centimeter: ", gm_per_cm3
    print "Hydrogen Atoms per Cubic Centimeter: ", nH_per_cm3
    print
    return [n_per_pc3, m_sol_per_pc3, gm_per_cm3, nH_per_cm3]


get_densities(n_sources=1000)




# Problem 5
def plot_inclinations(stepsize):
    plt.clf()

    def sample_inclinations(n_steps):
        inclinations, sin_cubed = [], []
        n = 0
        while n < n_steps:
            i = random.uniform(0, np.pi)
            sini = np.sin(i)
            inclinations.append(i)
            sin_cubed.append(sini**3)
            n += 1
        # return np.mean(sin_cubed)
        return (np.median(sin_cubed), np.mean(sin_cubed), inclinations)

    ns, means, medians, inclinations = [], [], [], []
    for n in np.arange(1, 5, stepsize):
        n_steps = 10**n
        ns.append(n_steps)
        median, mean, incls = sample_inclinations(n_steps)
        means.append(mean)
        medians.append(median)
        inclinations.append(incls)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw = {'width_ratios':[3, 1]})


    best_mean, best_median = means[-1], medians[-1]
    ax1.plot([ns[0], ns[-1]], [best_mean, best_mean], color='darkred',
             linewidth=4, linestyle='--', label='Best Mean Value')
    ax1.plot([ns[0], ns[-1]], [best_median, best_median], color='darkblue',
             linewidth=4, linestyle='--', label='Best Median Value')

    ax1.semilogx(ns, means, '-or', alpha=0.2)
    ax1.semilogx(ns, medians, '-ob', alpha=0.2)

    ax1.text(10**4, 1.07 * best_mean, 'Best Mean: ' + str(round(best_mean, 2)))
    ax1.text(10**4, 0.83 * best_median, 'Best Median: ' + str(round(best_median, 2)))
    ax1.set_xlabel('Number of Inclination Samples Drawn', weight='bold')
    ax1.set_ylabel(r"Mean Value of sin$^3 i$", weight='bold')
    ax1.legend()

    i_deg = np.array(inclinations[-1]) * 90/np.pi
    sns.distplot(i_deg, vertical=True, ax=ax2)
    ax2.set_ylabel('Inclination Angle (degrees)', weight='bold')
    ax2.set_xlabel('Sample Frequency', weight='bold')
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.linspace(start, end, 4))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    sns.despine()
    plt.tight_layout()

    plt.savefig('average_orbital_inclinations.png', dpi=200)
    plt.show()

    return i_deg




# The End
