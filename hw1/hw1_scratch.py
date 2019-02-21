import numpy as np
import pandas as pd
import astropy.units as u
from inspect import *
from astropy.coordinates import SkyCoord, Angle, get_constellation

d_sun_GC = 8122     # Sun/GC distance (parsec)



# Problem 3
def radec_to_galactic_astropy(coords):
    """
    Convert RA/dec coordinates to galactic (l, b) coordinates.

    Args: coords (tuple of strs): RA, dec values in a format understood
                                  by astropy.coordinates.Angle
    Returns: (l, b) tuple, in degrees.
    """
    ra_hms, dec_hms = Angle(coords[0]), Angle(coords[1])
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

# Test the handwritten converter against astropy's answers
vega_coords = ['18h 36m 56s', '+38d47m1s']
test_coords = ['04h37m48s', '-0d21m25s']
print radec_to_galactic(vega_coords)
print radec_to_galactic_astropy(vega_coords)




# Problem 4
def distance_to_galactic_center(galactic_coords, d):
    """
    Find the distance from the Galactic Center of an object at l, b, d.

    Args:
        galactic_coords (tuple of floats): l and b angles, in degrees.
        d (float) distance from Sun to star, in parsecs.
    """
    l, b = galactic_coords[0] * 3600, galactic_coords[1] * 3600
    h_star_gcp, d_star_sun = d * np.sin(b), d * np.cos(b)
    d_star_gc = np.sqrt(d_star_sun**2 + d_sun_GC**2 - 2*d_star_sun*d_sun_GC*np.cos(l))
    return d_star_gc

vega = radec_to_galactic_astropy(vega_coords)


class Star:
    """
    A Python class containing information about a fictional star defined
    by a given parameter set. Attributes include:

    - init_params: the initial list of parameters provided.
    - galactic_coords: the source's galactic coordinates.
    - pm_mag: the magnitude of the proper motion vector.
    - pm_posang: the position angle of the proper motion vector.
    - v_transverse: the source's transverse velocity.
    - v_space: the source's velocity relative to the Sun.
    - constellation: the constellation that the source is in.
    - d_from_GC: the source's distance from Galactic Center (parsecs)
    - closer: a boolean telling whether or not the source is
              closer to the Galactic Center than the Sun.

    To check the value of a single parameter (i.e. distance), print Star(params).distance
    To see a dataframe of all the values, pass print_df=True.


    """

    def __init__(self, params, print_df=True, print_help=False):
        """
        Get some info about a star from some other info.

        Args (unpacked from params):
            stellar_type (str): What kind of star it is (i.e. G2V)
            position (tuple of strs): RA, dec values in a format understood
                                          by astropy.coordinates.Angle
            parallax (float): parallax angle, in arcsecs
            proper_motion (tuple of floats): proper motion in RA/dec, in mas/year.
            rv (float): radial velocity, in km/s
        """
        stellar_type, position, parallax, proper_motion, v_radial = params
        self.init_params = params
        self.stellar_type = stellar_type
        self.proper_motion = proper_motion       # [mas/year, mas/year]
        self.distance = 1/parallax               # parsecs
        self.parallax = parallax                 # arcsecs
        self.position = position                 # [hms, dms]
        self.v_radial = v_radial                 # km/s

        self.galactic_coords = radec_to_galactic(self.position)  # degrees

        # Proper motion, described in Cartesian components
        self.pm_dec = self.proper_motion[1]
        # We don't need to scale by cos(dec) because the units are already in mas/year
        self.pm_ra = self.proper_motion[0] #* np.cos(self.pm_dec)

        # Proper motion, described in angular components
        self.pm_mag = np.sqrt(self.pm_ra**2 + self.pm_dec**2)        # mas/year
        # PA = angle east of north
        self.pm_posang = round(np.arctan(self.pm_ra/self.pm_dec), 4) # radians

        self.v_transverse = 4.74 * self.pm_mag * self.distance       # km/s

        # Space velocity is the third leg of the v_trans/v_rad triangle.
        self.v_space = np.sqrt(self.v_transverse**2 + self.v_radial**2)

        star_obj = SkyCoord(Angle(position[0]), Angle(position[1]), frame='icrs')
        self.constellation = get_constellation(star_obj)

        self.d_from_GC = self.distance_to_galactic_center()         # parsecs
        self.closer = True if self.d_from_GC > d_sun_GC else False

        d = [{'Name': 'Stellar Type', 'Value': self.stellar_type, 'units': 'N/A'},
             {'Name': 'Distance', 'Value': self.distance, 'units': 'parsec'},
             {'Name': 'Parallax', 'Value': self.parallax, 'units': 'arcsecs'},
             {'Name': 'Position', 'Value': self.position, 'units': '[hms, dms]'},
             {'Name': 'Galactic Coordinates', 'Value': self.galactic_coords,
              'units': 'degrees'},
             {'Name': 'Proper Motion (RA)', 'Value': self.pm_ra, 'units': 'mas/year'},
             {'Name': 'Proper Motion (Dec)', 'Value': self.pm_dec, 'units': 'mas/year'},
             {'Name': 'Proper Motion Magnitude', 'Value': self.pm_mag, 'units': 'mas/year'},
             {'Name': 'Proper Motion Position Angle', 'Value': self.pm_posang,
              'units': 'radians'},
             {'Name': 'Radial Velocity', 'Value': self.v_radial, 'units': 'km/s'},
             {'Name': 'Transverse Velocity', 'Value': self.v_transverse, 'units': 'km/s'},
             {'Name': 'Space Velocity', 'Value': self.v_space, 'units': 'km/s'},
             {'Name': 'Host Constellation', 'Value': self.constellation, 'units': 'N/A'},
             {'Name': 'Distance from Galactic Center', 'Value': self.d_from_GC,
              'units': 'parsecs'},
             {'Name': 'Closer than Sun to GC?', 'Value': self.closer, 'units': 'N/A'}
             ]

        self.full_param_df = pd.DataFrame(d)

        if print_help:
            print getdoc(self), '\n\n'

        if print_df:
            print self.full_param_df

    def distance_to_galactic_center(self):
        """
        Find the distance from the Galactic Center of an object at l, b, d.

        Args:
            galactic_coords (tuple of floats): l and b angles, in degrees.
            d (float) distance from Sun to star, in parsecs.
        """
        l, b = self.galactic_coords
        h_star_gcp = self.distance * np.sin(b)
        d_star_sun = self.distance * np.cos(b)
        d_star_gc = np.sqrt(d_star_sun**2 + d_sun_GC**2 - 2*d_star_sun*d_sun_GC*np.cos(l))
        return d_star_gc

# params = [stellar_type, position, parallax, proper_motion, rv]
params = ['G2V', ('04h24m46.0s', '12d37m22.0s'), 0.025, (-5.0, 24.0), 28.0]
prob4_star = Star(params, print_df=True, print_help=True)




# Problem 6
def sun_jup_com():
    # All units MKS
    m_jup, m_sol = 1.9e27, 2e30
    sma_jup = 7.78e11

    com = (m_jup * sma_jup)/(m_jup + m_sol)
    return com
