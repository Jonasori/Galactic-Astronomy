"""
Plot some cool stuff using some fun data.

Observational Techniques, Lab 3
Jonas Powell
November 19, 2018
"""

# Packages
import csv
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const
from sklearn.linear_model import LinearRegression
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia

# Constants
H0 = 72 * 1e-6                   # km s-1 pc-1
c = const.c.to('km/s').value     # km s-1


# PART 1
def plot_faber_jackson(save=False):
    """Plot the Faber-Jackson relationship.

    The Faber-Jackson relationship defines the relationship between luminosity
    and velocity dispersion for elliptical and S0 galaxies.
    """
    """
    The SQL call:
        SELECT TOP 10000
           p.petroMag_r, p.petroMagErr_r, p.extinction_r,
           p.petroMag_g, p.extinction_g,
           p.petroR50_r, p.petroR90_r, s.velDisp,
           p.dered_r, p.dered_g,
           p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
           p.run, p.rerun, p.camcol, p.field,
           s.specobjid, s.class, s.z as redshift,
           s.plate, s.mjd, s.fiberid
        FROM PhotoObj AS p
           JOIN SpecObj AS s ON s.bestobjid = p.objid
        WHERE
           p.petroR90_r/p.petroR50_r > 2.6
           AND class='galaxy'
           AND p.dered_g - p.dered_r> 1.0
           AND petroMag_r BETWEEN 0 and 19
           AND petroMagErr_r < 0.05
           AND s.z < 0.35
           AND veldisp > 30
           AND veldispErr/veldisp < 0.2
    """
    plt.close()

    # Read in the data
    path = '/Volumes/1TB Storage Drive/Desktop/masters/Observational-Techniques/lab3/'
    raw_query = path + 'SDSS_query.csv'
    new_path = path + 'SDSS_corrected.csv'
    with open(raw_query, 'rb') as inp, open(new_path, 'wb') as outp:
        writer = csv.writer(outp)
        for row in csv.reader(inp):
            if row != ['#Table1']:
                writer.writerow(row)
    df = pd.read_csv(new_path, sep=',')
    df['Abs. Magnitude'] = (df['r'] - df['extinction_r']) - 5 * \
        (np.log10(c * df['redshift'] / H0) - 1)
    # df.columns
    # len(df['petroR50_r'])

    len_test = len(df['Abs. Magnitude'])/2
    len_train = len(df['Abs. Magnitude']) - len_test
    train_x = df['Abs. Magnitude'][:len_train]
    test_x = df['Abs. Magnitude'][len_test:]
    train_y = df['velDisp'][:len_train]
    test_y = df['velDisp'][:len_test]

    train_x = train_x.reshape(len(train_x), 1)
    test_x = test_x.reshape(len(test_x), 1)
    train_y = train_y.reshape(len(train_y), 1)
    test_y = test_y.reshape(len(test_y), 1)

    regr = LinearRegression()
    regr.fit(train_x, train_y)

    predicted_y = regr.predict(df['Abs. Magnitude'].reshape(len(df['Abs. Magnitude']), 1))

    df.plot.scatter('Abs. Magnitude', 'velDisp',
                    c='petroMag_r', cmap='jet',
                    logy=True, alpha=0.2)

    plt.plot(df['Abs. Magnitude'], predicted_y, '.k')

    plt.xlabel('Absolute Magnitude', weight='bold')
    plt.ylabel('Velocity Dispersion', weight='bold')
    plt.title('Faber-Jackson Relation (SDSS Data)', weight='bold')
    plt.gca().invert_xaxis()
    if save is True:
        plt.savefig('faber-jackson.png', dpi=200)
    else:
        plt.show(block=False)
# It'd be nice to have this as a density plot


# plot_faber_jackson()


# PART 2
# Parameters by which to ID Pleiades objects.
pmra_lims, pmdec_lims = (12, 30), (-52, -40)
posra_lims, posdec_lims = (56.85 - 2, 56.85 + 2), (24.12 - 2, 24.12 + 2)
plx_lims = (5.5, 10)


# Compile the Hipparcos dataframe


def get_hipparcos_data():
    """Build a data structure for the Hipparcos data."""
    print "Building Hipparcos dataframe."

    coords = SkyCoord(ra='3h47m24s', dec='+24d7m0s',
                      unit=(u.deg, u.deg), frame='icrs')
    r = 2 * u.degree
    hip_results_raw = Vizier.query_region(coords, radius=r,
                                          catalog='I/239/hip_main')[0]
    hip_results = hip_results_raw.to_pandas()

    hip_df = pd.DataFrame()
    hip_df['Proper Motion (RA)'] = hip_results['pmRA']
    hip_df['Proper Motion (Dec)'] = hip_results['pmDE']
    hip_df['Distance'] = (hip_results['Plx'] * 1e-3)**(-1)
    hip_df['Parallax'] = hip_results['Plx']
    hip_df['Plx. Error'] = hip_results['e_Plx']
    hip_df['mag'] = hip_results['Vmag']
    hip_df['Color'] = hip_results['B-V']
    hip_df['Absolute Magnitude'] = hip_df['mag'] - \
        5 * (np.log10(hip_df['Distance']) - 1)
    hip_df['T Effective'] = [0] * len(hip_df['Color'])
    hip_df['Parallax'] = hip_results['Plx']
    hip_df['Plx. Error'] = hip_results['e_Plx']
    hip_df['Confidence'] = 1 - hip_results['e_Plx']/max(hip_results['e_Plx'])

    # Subset the data based on proper motion and Parallax cuts
    ra_cond1 = hip_results['pmRA'] > pmra_lims[0]
    ra_cond2 = hip_results['pmRA'] < pmra_lims[1]
    dec_cond1 = hip_results['pmDE'] > pmdec_lims[0]
    dec_cond2 = hip_results['pmDE'] < pmdec_lims[1]
    plx_cond1 = hip_results['Plx'] > plx_lims[0]
    plx_cond2 = hip_results['Plx'] < plx_lims[1]

    hip_df = hip_df[ra_cond1 & ra_cond2 & dec_cond1 &
                    dec_cond2 & plx_cond1 & plx_cond2]
    # hip_df.sort_values('Distance')
    pleiades_hip = {'Survey': 'Hipparcos',
                    'Mean Distance': round(np.mean(hip_df['Distance']), 1),
                    'Number of Stars': len(hip_df['Distance']),
                    'Data': hip_df}
    return pleiades_hip

pleiades_hip = get_hipparcos_data()


# Compile the GAIA dataframe
def get_gaia_data(n_sources=2000):
    """Build a data structure for the Gaia data.

    Paper on bands, colors, and equivalencies for GAIA:
    arxiv.org/pdf/1008.0815.pdf'

    Basically, since GAIA uses longer wavelength light, their colors don't map
    to UBVRI bandpasses (p. 4, Table 1). This paper refers us to use B-R for
    color, and since we used V magnitude when considering B-V color, I chose
    to use the R band magnitude for my apparent magnitude values.
    """
    print "Building GAIA dataframe for {} sources.".format(n_sources)

    gaia_str = "SELECT top {} * FROM gaiadr2.gaia_source \
                WHERE pmra between {} and {} \
                AND pmdec between {} and {} \
                AND ra between {} and {} \
                AND dec between {} and {} \
                AND parallax between {} and {} \
                AND parallax_error < 2 \
                AND parallax_over_error > 5 \
                ".format(n_sources,
                         pmra_lims[0], pmra_lims[1],
                         pmdec_lims[0], pmdec_lims[1],
                         posra_lims[0], posra_lims[1],
                         posdec_lims[0], posdec_lims[1],
                         plx_lims[0], plx_lims[1])

    job = Gaia.launch_job(gaia_str)  # , dump_to_file=True)
    # job = Gaia.launch_job(gaia_str)
    gaia_results_raw = job.get_results()
    # gaia_results_raw['phot_rp_mean_mag'].description

    gaia_results = gaia_results_raw.to_pandas()
    # gaia_cols = sorted(gaia_results.columns)

    print "Acquired data; now building dataframe..."
    gaia_df = pd.DataFrame()
    gaia_df['Distance'] = (gaia_results['parallax'] * 1e-3)**(-1)
    gaia_df['Proper Motion (RA)'] = gaia_results['pmra']
    gaia_df['Proper Motion (Dec)'] = gaia_results['pmdec']
    gaia_df['mag'] = gaia_results['phot_rp_mean_mag']
    gaia_df['Color'] = gaia_results['bp_rp']
    gaia_df['Absolute Magnitude'] = gaia_df['mag'] - \
        5 * (np.log10(gaia_df['Distance']) - 1)
    gaia_df['T Effective'] = gaia_results['teff_val']
    gaia_df['Parallax'] = gaia_results['parallax']
    gaia_df['Plx. Error'] = gaia_results['parallax_error']
    gaia_df['Confidence'] = 1 - gaia_results['parallax_error']/max(gaia_results['parallax_error'])

    pleiades_gaia = {'Survey': 'Gaia',
                     'Mean Distance': round(np.mean(gaia_df['Distance']), 1),
                     'Number of Stars': len(gaia_df['Distance']),
                     'Data': gaia_df,
                     'Full Results': gaia_results,
                     'Full Table': gaia_results_raw}

    return pleiades_gaia

pleiades_gaia = get_gaia_data()




# Compile the GAIA dataframe
def get_local_data(n_sources=10000):
    """Build a data structure for the Gaia data.

    Paper on bands, colors, and equivalencies for GAIA:
    arxiv.org/pdf/1008.0815.pdf'

    Basically, since GAIA uses longer wavelength light, their colors don't map
    to UBVRI bandpasses (p. 4, Table 1). This paper refers us to use B-R for
    color, and since we used V magnitude when considering B-V color, I chose
    to use the R band magnitude for my apparent magnitude values.
    """
    print "Building local dataframe for {} sources.".format(n_sources)

    local_str = "SELECT top 20000 * FROM gaiadr2.gaia_source \
                WHERE parallax between 1 and 20"

    job = Gaia.launch_job(local_str)  # , dump_to_file=True)
    # job = Gaia.launch_job(gaia_str)
    local_results_raw = job.get_results()
    # local_results_raw['phot_rp_mean_mag'].description

    local_results = local_results_raw.to_pandas()
    # local_cols = sorted(local_results.columns)

    print "Acquired data; now building dataframe..."
    local_df = pd.DataFrame()
    local_df['Distance'] = (local_results['parallax'] * 1e-3)**(-1)
    local_df['Proper Motion (RA)'] = local_results['pmra']
    local_df['Proper Motion (Dec)'] = local_results['pmdec']
    local_df['mag'] = local_results['phot_rp_mean_mag']
    local_df['Color'] = local_results['bp_rp']
    local_df['Absolute Magnitude'] = local_df['mag'] - \
        5 * (np.log10(local_df['Distance']) - 1)
    local_df['T Effective'] = local_results['teff_val']
    local_df['Parallax'] = local_results['parallax']
    local_df['Plx. Error'] = local_results['parallax_error']
    local_df['Confidence'] = 1 - local_results['parallax_error']/max(local_results['parallax_error'])

    pleiades_local = {'Survey': 'local',
                      'Mean Distance': round(np.mean(local_df['Distance']), 1),
                      'Number of Stars': len(local_df['Distance']),
                      'text_loc1': (1.1, -2.2),
                      'text_loc2': (2, -0.2),
                      'Data': local_df,
                      'Full Results': local_results,
                      'Full Table': local_results_raw}

    return pleiades_local


local_sources = get_local_data()




# A generic plotter for the two datasets
def make_plots(survey, save=False):
    """Make the necessary plots for each dataset.

    Note that coloring is done basically by the inverse of error, given by:
    confidence = 1 - (sigma/sigma_max)
    """
    plt.close()
    print "Beginning plot-making process for " + survey['Survey'] + "..."

    df = survey['Data']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    df.plot.scatter('Proper Motion (RA)', 'Proper Motion (Dec)',
                    ax=axes[0], marker='.', c='Confidence',
                    cmap='jet', colorbar=False, alpha=0.5
                    )

    # ylims = (max(df['Absolute Magnitude']), min(df['Absolute Magnitude']))
    df.plot.scatter('Color', 'Absolute Magnitude',
                    ax=axes[1], marker='.',
                    c='Confidence', cmap='jet', alpha=0.5
                    )

    text_str1 = 'Number of stars: ' + str(survey['Number of Stars'])
    text_str2 = 'Mean distance:\n' + str(survey['Mean Distance']) + ' parsecs'

    pmra = df['Proper Motion (RA)']
    pmdec = df['Proper Motion (Dec)']
    text_loc1_0 = max(pmra) - 0.6 * (max(pmra) - min(pmra))
    text_loc1_1 = max(pmdec) - 0.05 * (max(pmdec) - min(pmdec))
    text_loc1 = (text_loc1_0, text_loc1_1)

    abs_mag = df['Absolute Magnitude']
    color = df['Color']
    text_loc2_0 = max(color) - 0.4 * (max(color) - min(color))
    text_loc2_1 = min(abs_mag) + 0.1 * (max(abs_mag) - min(abs_mag))
    text_loc2 = (text_loc2_0, text_loc2_1)

    print text_loc1
    print text_loc2

    axes[0].annotate(text_str1, text_loc1, weight='bold')
    axes[1].annotate(text_str2, text_loc2, weight='bold')
    axes[1].set_ylim(axes[1].get_ylim()[::-1])

    axes[1].set_xticks(np.linspace(min(df['Color']), max(df['Color']), 4))
    axes[1].set_yticks(np.linspace(min(df['Absolute Magnitude']),
                                   max(df['Absolute Magnitude']), 4))

    axes[0].set_xticks(np.linspace(min(df['Proper Motion (RA)']),
                                   max(df['Proper Motion (RA)']), 4))
    axes[0].set_yticks(np.linspace(min(df['Proper Motion (Dec)']),
                                   max(df['Proper Motion (Dec)']), 4))

    axes[0].set_title('Proper Motions', weight='bold')
    axes[1].set_title('HR Diagram', weight='bold')
    fig.suptitle('Pleiades Cluster Characteristics, from ' + survey['Survey'] + '\n\n',
                 weight='bold', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.1)
    if save is True:
        outname = './pleiades_' + survey['Survey'] + '.png'
        plt.savefig(outname, dpi=200)
        print "Saved to ", outname
    else:
        print "Showing:"
        plt.show(block=False)


# make_plots(pleiades_gaia, save=False)


# The End
