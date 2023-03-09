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
import glob # this package lets you search for filenames

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





""" STAR FLUX PERIOD GRAPH START """
plt.figure()
fname = 'FrontS017978.csv' # put your filename here

data = pd.read_csv(ddir+fname) # load in CSV data as a Pandas object
print(data.keys()) # see what's in it
time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables
dt = np.median(np.diff(time))
print('Nyquist Limit',0.5/dt,'cycles per hour') # can't get frequencies higher than the Nyquist limit

plt.title('Period Luminosity of Star ')
plt.plot(time,flux,'.',markersize=16)
plt.xlabel('Time (h)')
plt.ylabel('Relative Flux')
""" STAR FLUX PERIOD GRAPH END """




""" LOMB-SCARGLE GRAPH START """
plt.figure()
LS = LombScargle(time,flux) # initialize a Lomb-Scargle algorithm from Astropy
freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
power = LS.power(freqs) # calculate LS power

print('Best period: %.2f h' % (1/freqs[np.argmax(power)]))

plt.plot(freqs,power)
plt.xlabel('Frequency (c/h)')
plt.ylabel('LS Power')
""" LOMB-SCARGLE GRAPH END """





""" LOOPING OVER STARS IN THE DIRECTORY TO OBTAIN PERIODOGRAMS START """
# actual periodograms can be obtained with some changes to the code
# be careful not to plot 1 million plots for every star
plt.figure()
fnames = glob.glob(ddir+'*.csv')
freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
periods = [] # start an empty list to hold the period 
names = []

for fname in tqdm(fnames): # tqdm is a package that gives you a progress bar - neat! 
    data = pd.read_csv(fname) # load in CSV data as a Pandas object

    time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables

    LS = LombScargle(time,flux) # initialize a Lomb-Scargle
    power = LS.power(freqs) # calculate LS power 
    bestfreq = freqs[np.argmax(power)] # which frequency has the highest Lomb-Scargle power?
    
    pred = LS.model(time,bestfreq) # make a sine wave prediction at the best frequency
    
    periods.append(1/bestfreq) # add each period to the list
    names.append(os.path.basename(fname).strip('.csv')) # os.path.basename gets rid of directories and gives you the filename; then we strip '.csv'
    
periods = np.array(periods) # turn it from a list to an array
""" LOOPING OVER STARS IN THE DIRECTORY TO OBTAIN PERIODOGRAMS END """





""" HR DIAGRAM WITH VARIABLE STARS START """
variables = pd.DataFrame({'Name':names, 'Period':periods})
               # you can turn a dictionary into a dataframe like this
variables.Name = variables.Name.astype('|S') # have to do this so that it knows the names are strings

all_star_files = glob.glob(ddir_stars+'/*/Star_Data.csv') 

all_stars = pd.concat([pd.read_csv(table) for table in all_star_files]) # we are concatenating a list of dataframes; 
#we generate this list with a "list comprehension", a loop you write inside a list bracket 

all_stars.Name = all_stars.Name.astype('|S') # have to do this so that it knows the names are strings
all_stars = all_stars[all_stars.Parallax > 0.01] # 10 mas parallax cut
print(len(all_stars),'stars above 10 mas parallax') # check how many stars there are total with good parallax

variables = pd.merge(all_stars,variables,on='Name') # merge these two arrays according to the keyword 'name'
print('Of which',len(variables),'variables') # cut down to a small list

m0, m1, m2 = np.log10(all_stars['BlueF']), np.log10(all_stars['GreenF']), np.log10(all_stars['RedF']) 
colour = m2-m0
abs_mag = m1 + 2*np.log10(1./all_stars.Parallax) 

v0, v1, v2 = np.log10(variables['BlueF']), np.log10(variables['GreenF']), np.log10(variables['RedF']) 
variable_colour = v2-v0
abs_mag_v = v1 + 2*np.log10(1./variables.Parallax)

s = plt.plot(colour,abs_mag,'.C0')
h = plt.plot(variable_colour,abs_mag_v,'.C2',marker='*',markersize=10)

plt.legend([s, h],['Steady','Variable'])
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')  
""" HR DIAGRAM WITH VARIABLE STARS END """





""" PERIOD LUMINOSITY DIAGRAM START """
plt.figure()
plt.plot(variables.Period,abs_mag_v,'.',color='C2')
plt.xlabel('Period (h)')
plt.ylabel('Log Flux');
""" PERIOD LUMINSOITY DIAGRAM END """

