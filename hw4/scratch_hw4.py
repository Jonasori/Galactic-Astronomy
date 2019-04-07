"""
Galactic Astronomy: Problem Set 4
Jonas Powell
April 4, 2019


Cool thing: dir(object) returns all that object's attributes and methods.
"""

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", module='astropy.io.votable.tree')
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import datashader as ds
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from numba import jit
from importlib import reload
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from astroquery.gaia import Gaia

from get_cluster import Cluster







m5 = Cluster('m5', n_sources=7000)

m5.make_plots(save=True, d_color=0.2, d_mag=-0.15, iso_min=35, iso_max=None)
# m5.plot_HR_isochrones(d_color=0.3, d_mag=0., iso_min=30, iso_max=40)



m43 = Cluster('m43', n_sources=7000)
m43.make_plots(save=True, d_color=0.5, d_mag=0.5, iso_min=0, iso_max=10)
# m43.plot_HR_isochrones(d_color=0., d_mag=-1.6, iso_min=0, iso_max=10)




m42 = Cluster('m42', n_sources=7000)
m42.make_plots(save=True, d_color=0., d_mag=-1.1, iso_min=0, iso_max=10)
# m42.plot_HR_isochrones(d_color=0., d_mag=-1.6, iso_min=0, iso_max=5)




ngc104 = Cluster('ngc104', n_sources=7000)
ngc104.make_plots(save=True, d_color=0.2, d_mag=0., iso_min=35, iso_max=None)
# ngc104.plot_HR_isochrones(d_color=1., d_mag=3, iso_min=None, iso_max=None)




m67 = Cluster('m67', n_sources=7000)
m67.make_plots(save=True, d_color=0.9, d_mag=1.8, iso_min=0, iso_max=10)





m44 = Cluster('m44', n_sources=7000)
m44.make_plots(save=False, d_color=0.9, d_mag=1.8, iso_min=0, iso_max=2)







# The End
