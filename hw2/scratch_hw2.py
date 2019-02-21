"""
Calculations for Problem Set 2
Galactic Astronomy
Due Feb. 14
"""

import sys
import emcee
import cPickle
import numpy as np
import pandas as pd
import seaborn as sns
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle, get_constellation
from astropy.constants import c, h, k_B
from collections import Counter
from inspect import *
sns.set_style('white')

# c, h, k = c.to('cm/s').value, h.to('erg s').value, k_B.to('erg/K').value
# c, h, k = c.to('m/s').value, h.to('J s').value, k_B.to('J/K').value




# Problem 2

def spectrum(t, coeff, show=False):

    temp = t * u.K
    mags = [9.31, 8.94, 8.11, 7.93, 7.84]
    wavelengths = [0.438 * u.um, 0.545 * u.um, 1.22 * u.um, 1.63 * u.um, 2.19 * u.um]
    freqs = [(c/(lam).decompose()).to('Hz') for lam in wavelengths] # Hz

    unit_nu = (1e-20 * u.erg / (u.cm**2 * u.s * u.Hz)).to('Jy')
    zpfs_nu = [4.063 * unit_nu, 3.636 * unit_nu, 1.589 * unit_nu, 1.021 * unit_nu, 0.64 * unit_nu]
    data_f_nu = [zpf * 10**(-0.4*mag) for mag, zpf in zip(mags, zpfs_nu)]

    fudge = (coeff * 1e18*u.Jy)**(-1)
    model_f_nu = [((2 * h * nu**3 * c**(-2)) / (-1 + np.exp(h * nu/(k_B * temp)))).to('Jy') * fudge
                 for nu in freqs]

    freqs_thz = [nu.to('THz').value for nu in freqs]
    model = [f.value for f in model_f_nu]
    data = [f.value for f in data_f_nu]


    if show:
        plt.show()
    else:
        pass

    return (freqs_thz, model, data)

m, d = spectrum(6550, 4.8, save=True)




def lnprob(p, mags, priors, save=False):
    t, coeff = p

    # Check on priors:
    for param, prior in zip(p, priors):
        if not prior['min'] < param < prior['max']:
            return -np.inf

    temp = t * u.K
    wavelengths = [0.438 * u.um, 0.545 * u.um, 1.22 * u.um, 1.63 * u.um, 2.19 * u.um]
    freqs = [(c/(lam).decompose()).to('Hz') for lam in wavelengths] # Hz


    unit_nu = (1e-20 * u.erg / (u.cm**2 * u.s * u.Hz)).to('Jy')
    zpfs_nu = [4.063 * unit_nu, 3.636 * unit_nu, 1.589 * unit_nu, 1.021 * unit_nu, 0.64 * unit_nu]
    data_f_nu = [zpf * 10**(-0.4*mag) for mag, zpf in zip(mags, zpfs_nu)]

    fudge = (coeff * 1e18*u.Jy)**(-1)
    model_f_nu = [((2 * h * nu**3 * c**(-2)) / (-1 + np.exp(h * nu/(k_B * temp)))).to('Jy') * fudge
                 for nu in freqs]

    freqs4plotting = [nu.to('THz').value for nu in freqs]
    model = np.array([f.value for f in model_f_nu])
    data = np.array([f.value for f in data_f_nu])

    chisq = np.sum((model - data)**2)
    lnp = -0.5 * chisq
    return lnp





mags = [9.31, 8.94, 8.11, 7.93, 7.84]
def run_mcmc(mags=mags, nsteps=1000):

    ndim, nwalkers = 2, 30
    priors_t = {'min': 0, 'max': 20000}
    priors_coeff = {'min': 0, 'max': 30}
    priors = [priors_t, priors_coeff]

    p0 = np.random.normal(loc=(4000, 4), size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[mags, priors])
    pos, prob, state = sampler.run_mcmc(p0, 10)
    prob
    sampler.reset()
    print "Finished burn-in; starting full run now."
    # sampler.run_mcmc(pos, nsteps)

    run = sampler.sample(pos, iterations=nsteps, storechain=True)
    steps = []
    for i, result in enumerate(run):
        # Maybe do this logging out in the lnprob function itself?
        pos, lnprobs, blob = result

        new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
        steps += new_step

        sys.stdout.write("Completed step {} of {}  \r".format(i, nsteps) )
        sys.stdout.flush()

    df = pd.DataFrame(steps)
    df.columns = ['temp', 'coeff', 'lnprob']

    print "Finished MCMC."
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    # xlims, ylims = (priors_t['min'], priors_t['max']), (priors_coeff['min'], priors_coeff['max'])


    return (sampler, df)


# run2 = run_mcmc(mags, nsteps=50)



def mcmc_full_driver(prev_run=None, nsteps=20, save=False):
    plt.close()

    # First, execute the MCMC run. If one is passed as an argument, don't.
    if not prev_run:
        sampler, df = run_mcmc(nsteps=nsteps)
    else:
        sampler, df = prev_run

    # Now make some plots.
    jp = sns.jointplot(x=sampler.flatchain[:,0], y=sampler.flatchain[:,1],
                       kind="kde")
    jp.set_axis_labels('Effective Temperature', r"$\Omega_0$", weight='bold')
    plt.tight_layout()
    if save:
        plt.savefig('prob2_kde.pdf')
        print "Saved plot to prob2_kde.pdf"
    else:
        plt.show()


    # Get the data for the SED.
    t = df[df['lnprob'] == max(df['lnprob'])]['temp'].values[0]
    coeff = df[df['lnprob'] == max(df['lnprob'])]['coeff'].values[0]
    freqs, model, data = spectrum(t, coeff, show=False)
    print "Got spectrum."

    plt.plot(freqs, model, 'or', label='Model')
    plt.plot(freqs, data, 'ob', label='Data')
    plt.plot(freqs, model, '-r')
    plt.plot(freqs, data, '-b')

    plt.xlabel('Frequency (THz)', weight='bold')
    plt.ylabel(r"F$_{\nu}$ (Jy)", weight='bold')

    plt.title("Best-Fit SED at T = {}".format(round(t, 2)), weight='bold')
    plt.legend()
    plt.tight_layout()
    sns.despine()

    if save:
        plt.savefig('prob2_spectrum.pdf')
        print "Saved plot to prob2_spectrum.pdf"
    else:
        plt.show()

mcmc_full_driver()

# pos, prob, state = run
#
# run.chain.shape
#
# run.flatchain
#
# run.get_lnprob(run.chain[0, 0, :])


# Problem 3

test_coords = ['04h37m48s', '-0d21m25s']

pos_radec = ('13h42m25.6s', '-7d13m42.1s')
def radec_to_galactic_astropy(pos_radec):
    """
    Convert RA/dec coordinates to galactic (l, b) coordinates.

    Args: coords (tuple of strs): RA, dec values in a format understood
                                  by astropy.coordinates.Angle
    Returns: (l, b) tuple, in degrees.
    """
    ra_hms, dec_hms = Angle(pos_radec[0]), Angle(pos_radec[1])
    radec_coords_deg = SkyCoord(ra=ra_hms, dec=dec_hms, frame='icrs')
    galactic_coords_str = radec_coords_deg.transform_to('galactic').to_string()
    galactic_coords_degs = [float(coord) for coord in galactic_coords_str.split(' ')]
    return galactic_coords_degs



def radec_to_galactic(coords):
    """
    Convert RA/dec coordinates to galactic (l, b) coordinates by hand.

    Sources:
        NGP coords taken from:
        https://en.wikipedia.org/wiki/Galactic_coordinate_system

        Conversion formula adapted from:
        http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_coords.php
    Args:
        coords (tuple of strs): RA, dec values in a format understood
                                      by astropy.coordinates.Angle
    Returns: (l, b) tuple, in degrees.
    """

    def gross_coords_to_rads(coords):
        ra, dec = coords
        coords = SkyCoord(ra=ra, dec=dec, frame='icrs')
        ra_rad, dec_rad = [float(a) * np.pi/180
                           for a in coords.to_string().split()]
        return (ra_rad, dec_rad)

    ra, dec = gross_coords_to_rads(coords)
    ra_NGP, dec_NGP = gross_coords_to_rads(['12h51m26.00s', '+27d 7m 42.0s'])
    l_NCP = 122.93 * np.pi/180

    b = np.arcsin(np.sin(dec_NGP) * np.sin(dec) \
                  + np.cos(dec_NGP) * np.cos(dec) \
                  * np.cos(ra - ra_NGP))

    x1 = np.cos(dec) * np.sin(ra - ra_NGP)
    x2 = np.cos(dec_NGP) * np.sin(dec) \
         - np.sin(dec_NGP) * np.cos(dec) * np.cos(ra - ra_NGP)

    # Arctan2 is basically a smart version of arctan(x1/x2)
    l = l_NCP - np.arctan2(x1, x2)

    # Convert to degrees and round out to 4 decs for prettiness.
    l, b = round(l * 180/np.pi, 4), round(b * 180/np.pi, 4)
    return [l, b]













# Kepler populations

def plot_planet_pops_by_stellar_type(cmap='inferno_r', spectral_grouping=False, save=True):

    # cmaps: viridis, Spectral, ocean, inferno

    host_stars_df = pd.read_csv('plotting_data.csv')

    if spectral_grouping:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.countplot(x=host_stars_df['spectral_group'], palette=cmap, ax=ax)

        # sns.countplot(x=host_stars_df['spectral_group'], palette='ocean', ax=ax2)

        ax.legend().remove
        ax.set_ylabel('Number of Planets', weight='bold')
        ax.set_xlabel('Spectral Type', weight='bold')
        ax.set_title('Planet Populations by Spectral Type from Kepler', weight='bold')
        sns.despine()

        if save:
            plt.savefig('planet_dist_by_spectral_group.pdf')
        else:
            plt.show()
    else:
        labs = []
        for id in sorted(list(set(host_stars_df['spt_id']))):
            spt_name = host_stars_df[host_stars_df['spt_id'] == id]['spt'].values[0]
            labs.append(spt_name)

        fig, ax = plt.subplots(figsize=(14, 4))
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        sns.countplot(x=host_stars_df['spt_id'], palette=cmap, ax=ax)

        ax.legend().remove
        ax.set_ylabel('Number of Planets', weight='bold')
        ax.set_xlabel('Spectral Type', weight='bold')
        ax.set_title('Planet Populations by Spectral Type from Kepler', weight='bold')
        ax.set_xticklabels(labs, rotation=45)

        sns.despine()

        if save:
            plt.savefig('planet_dist_by_spt.pdf')
        else:
            plt.show()


# Counter(host_stars_df['spt'])
# host_stars_df
# sorted(Counter(host_stars_df['spt']).values())
# host_stars_df['spt_id']/Counter(host_stars_df['spt_id']).values()


# stellar_info = pd.read_csv('spectraltype_data.csv')
# stellar_info[stellar_info['SpT'] == 'G2V']['Mv']











# The End
