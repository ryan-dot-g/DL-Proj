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
from PyAstronomy import pyasl #must install this astro package

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
star_name = fname[:-4]

data = pd.read_csv(ddir+fname) # load in CSV data as a Pandas object
print(data.keys()) # see what's in it
time, flux = data.Time, data.NormalisedFlux # just extract the columns as variables
dt = np.median(np.diff(time))
print('Nyquist Limit',0.5/dt,'cycles per hour') # can't get frequencies higher than the Nyquist limit

plt.title('Period Luminosity of Star' + star_name)
plt.plot(time,flux,'.',markersize=16)
plt.xlabel('Time (h)')
plt.ylabel('Relative Flux')
plt.show()
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
plt.title('LS Periodogram for star ' + star_name)
plt.show()
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
plt.title('HR Diagram of all Stars')
plt.show()
# stars denoted by * in the HR diagram are variable
""" HR DIAGRAM WITH VARIABLE STARS END """





""" PERIOD LUMINOSITY DIAGRAM START """
plt.figure()
plt.plot(variables.Period,abs_mag_v,'.',color='C2')
plt.xlabel('Period (h)')
plt.ylabel('Log Flux');
plt.title('Period-Luminosity Diagram of all Variable Stars')
plt.show()
# there should be two distinct sections of variable stars, usually short and long period
""" PERIOD LUMINSOITY DIAGRAM END """



""" DISTANCES TO VARIABLE STARS START """
variables["Dist"] = 1/variables.Parallax # distance = 1/parallax where parallax in arcseconds
""" DISTANCES TO VARIABLE STARS END """



""" LINEAR FITTING TO PERIOD LUMINOSITY START """
variables["abs_mag_v"] = abs_mag_v
# we define class 1 stars to have 20<P<30 hours and class 2 P>30 hours
# there is out outlier which doesnt fall into any class, but these class definitions
# remove that outlier
class1 = variables[(20 < variables.Period) & (variables.Period < 30)]
class2 = variables[variables.Period > 30]


A1 = np.vander(class1.Period,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b1, residuals1, rank1, s1 = np.linalg.lstsq(A1,class1.abs_mag_v)
print('Recovered parameters C1: %.2f, %.2f' % (b1[0],b1[1]))
print('C1 equation is Log Flux  =', b1[0], 'P +', b1[1] )
reconstructed1 = A1 @ b1 # @ is shorthand for matrix multiplication in python

A2 = np.vander(class2.Period,2) # the Vandermonde matrix of order N is the matrix of polynomials of an input vector 1, x, x**2, etc
b2, residuals2, rank2, s2 = np.linalg.lstsq(A2,class2.abs_mag_v)
print('Recovered parameters C2: %.2f, %.2f' % (b2[0],b2[1]))
print('C2 equation is Log Flux  =', b2[0], 'P +', b2[1] )
reconstructed2 = A2 @ b2 # @ is shorthand for matrix multiplication in python

plt.plot(variables.Period,abs_mag_v,'.',color='C2')
plt.plot(class1.Period,reconstructed1,'-b',label='Reconstructed C1')
plt.plot(class2.Period,reconstructed2,'-r',label='Reconstructed C2')

plt.legend()
plt.xlabel('Period (h)')
plt.ylabel('Log Flux');
plt.title('Period-Luminosity Diagram of all Variable Stars with Directly Determinable Distance')
plt.show()
""" LINEAR FITTING TO PERIOD LUMINOSITY END """



""" FINDING DISTANCE TO ALL VARIABLE STARS START """

all_variables = pd.DataFrame({'Name':names, 'Period':periods})
               # you can turn a dictionary into a dataframe like this
               # these are ALL the variable stars

all_stars = pd.concat([pd.read_csv(table) for table in all_star_files]) # we are concatenating a list of dataframes; 
#we generate this list with a "list comprehension", a loop you write inside a list bracket 

all_variables = pd.merge(all_stars,all_variables,on='Name') # merge these two arrays according to the keyword 'name'
    # we merge to make sure every variable star has basic info from all_stars
    # such as coordinate and radial velocity
    
# now we find the absolute magnitude of each star based off the period and the 
# already determined period-luminosity relations    
all_variables_c1 = all_variables[(20 < all_variables.Period) & (all_variables.Period < 30)]
all_variables_c2 = all_variables[(40 < all_variables.Period) & (all_variables.Period < 50)]

all_variables_c1["Absolute_Magnitude"] = b1[0] * all_variables_c1.Period + b1[1]
all_variables_c2["Absolute_Magnitude"] = b2[0] * all_variables_c2.Period + b2[1]

plt.scatter(all_variables_c1.Period, all_variables_c1.Absolute_Magnitude)
plt.scatter(all_variables_c2.Period, all_variables_c2.Absolute_Magnitude)

# here we rearrange the dist, absmag, flux formula as written in """HR diagram with variable stars"""
# to determine distance: #abs_mag_v = v1 + 2*np.log10(1./variables.Parallax)
all_variables_c1["Dist"] = 10**((all_variables_c1["Absolute_Magnitude"] - np.log10(all_variables_c1["GreenF"])) / 2)
all_variables_c2["Dist"] = 10**((all_variables_c2["Absolute_Magnitude"] - np.log10(all_variables_c2["GreenF"])) / 2)

""" FINDING DISTANCE TO ALL VARIABLE STARS END """

# example plot to show realtionship
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.scatter(all_variables_c1["Dist"], all_variables_c1["GreenF"])

