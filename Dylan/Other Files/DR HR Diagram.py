# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:37:35 2023

@author: dylan
"""

import numpy as np
import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from astropy.timeseries import LombScargle
from astropy.table import Table

### STAR POSITION PLOTTER START ###

# configure notebook for plotting
# %pylab inline --no-import-all 
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme
colours = mpl.rcParams['axes.prop_cycle'].by_key()['color'] # allows access to colours
# subsequent lines default plot settings
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
matplotlib.rcParams['savefig.dpi']= 300             #72 

import warnings
warnings.filterwarnings('ignore')

ddir = 'C:/Users/dylan/Documents/GitHub/DL-Proj/DATA'
camera = '/Front'
stars = Table.read(ddir + camera + '/Star_Data.csv', format='ascii') 

plt.figure()
plt.title('Star Positions')
plt.scatter(stars['X'],stars['Y'])
plt.xlabel('x (pix)')
plt.ylabel('y (pix)');

### STAR POSITION PLOTTER END ###



### HR DIAGRAM START ###

m0, m1, m2 = (np.log10(stars['BlueF']), 
              np.log10(stars['RedF']), 
              np.log10(stars['GreenF'])) 
colour = m2-m0

plt.figure()
plt.title('HR Diagram')
s = plt.scatter(colour,m1)
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')

### HR DIAGRAM END ###

