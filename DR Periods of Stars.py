# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:19:15 2023

@author: dylan
"""

import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm # tqdm is a package that lets you make progress bars to see how a loop is going
import os 
import pandas as pd # pandas is a popular library in industry for manipulating large data tables
from astropy.timeseries import LombScargle

# configure notebook for plotting
# %matplotlib inline

mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme

# subsequent lines default plot settings
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)   
matplotlib.rcParams['font.size']=16              
matplotlib.rcParams['savefig.dpi']= 300             

import warnings
warnings.filterwarnings('ignore')

ddir_stars = 'C:/Users/dylan/Documents/GitHub/DL-Proj/DATA' # point this to where you unzip your data!
ddir = ddir_stars + '/Variable_Star_Data/'





### STAR FLUX PERIOD GRAPH START ###

fname = 'FrontS017978.csv' # put your filename here

data = pd.read_csv(ddir+fname) # load in CSV data as a Pandas object
print(data.keys()) # see what's in it
time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables
dt = np.median(np.diff(time))
print('Nyquist Limit',0.5/dt,'cycles per hour') # can't get frequencies higher than the Nyquist limit

plt.title('Period Luminosity of Star')
plt.plot(time,flux,'.',markersize=16)
plt.xlabel('Time (h)')
plt.ylabel('Relative Flux')

### STAR FLUX PERIOD GRAPH END ###


### LOMB-SCARGLE GRAPH START ###

LS = LombScargle(time,flux) # initialize a Lomb-Scargle algorithm from Astropy
freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
power = LS.power(freqs) # calculate LS power

print('Best period: %.2f h' % (1/freqs[np.argmax(power)]))

plt.plot(freqs,power)
plt.xlabel('Frequency (c/h)')
plt.ylabel('LS Power')

### LOMB-SCARGLE GRAPH END ###
