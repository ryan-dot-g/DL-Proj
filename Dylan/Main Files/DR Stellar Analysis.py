# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:43:11 2023

@author: dylan
"""

import numpy as np # for maths 
from tqdm import tqdm # status bar
import glob # directories
import pandas as pd # tables and data frames
from matplotlib import pyplot as plt
from scipy import integrate # numerical integration

## note that if you run this program it will save a csv in a folder called 'Dylan'
    ## see bottom for more detail`

## i still need to do UNCERTAINTIES!!!

# constants
h = 6.626 * 10**-34 
c = 3 * 10**8 
k = 1.38064 * 10**-23 
stef_boltz = 5.67 * 10**-8
parallaxCutoff = 0.01

# ddir = 'C:/Users/dylan/Documents/GitHub/DL-Proj/DATA/'
ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\'
all_stars = glob.glob(ddir + '*/Star_Data.csv') # all star data
allStarsDf = pd.concat( pd.read_csv(catalog) for catalog in all_stars ) 
    # super dataframe containing all stars
goodStars = allStarsDf[allStarsDf.Parallax > parallaxCutoff] # calibrate with star if sufficient parallax
goodStars["m0"], goodStars["m1"], goodStars["m2"] = (np.log10(goodStars.BlueF),
                                                     np.log10(goodStars.GreenF),
                                                     np.log10(goodStars.RedF)) # different color fluxes
goodStars["colour"] = goodStars.m2 - goodStars.m0
goodStars["dist"] = 1/goodStars.Parallax # distance = 1/parallax where parallax in arcseconds
goodStars["abs_mag"] = goodStars.m1 + 2 * np.log10(goodStars.dist) # absolute magnitude using log scale
    # this is to find stars which we have a distance to



# wavelength for each filter (blue green red) in nm converted to m
xdata = np.array([440,500,700]) * 10**-9
#ydata = np.array([1.39E-14,	9.17E-15,	2.99E-15])

# after constants, we define functions to begin

def bbcurve(wavelength, temperature):
    """
    this is just Planck's Law (for spectral radiance and blackbody)'
    """
    return (2*h*c**2) / (wavelength**5 * ( np.exp((h*c)/(wavelength*k*temperature)) - 1))


def bbcurve_comparison(temperature, BlueF, GreenF, RedF):
    """ 
    this function finds the approx bbcurve using some photometric data and a
    stellar temperature and compares it to the shape of the photometric data
    
    the likeness in shape between the bbcurve and the photometry data is saved 
    in a variable called 'equality', with 3 representing the most alike
    """
    spec_rad = bbcurve(xdata, temperature)
    ydata = [BlueF, GreenF, RedF]
    constants = np.divide(spec_rad, ydata)
    quo1 = max((constants[1] / constants[0]), (constants[0] / constants[1]))
    quo2 = max((constants[2] / constants[0]), (constants[0] / constants[2]))
    quo3 = max((constants[2] / constants[1]), (constants[1] / constants[2]))
    equality = quo1 + quo2 + quo3
    return equality


def find_star_temp(BlueF, GreenF, RedF):
    """
    this function finds star temp by using previoulsy defined functions
    """
    starttemp = 100
    endtemp = 40100 # in our universe stars could reach 100 000 K, this data set 
                    # shows max temp of around 36000K
    jump = 100
    amount_vals = (endtemp-starttemp) / jump
    equality_list = [] #np.ones(starttemp-25) * 10**100
    # this section especially can be optimised to reduce computational time
    # consider ending computation early bsaed on numerical gradient of function
    for n in range(0, int(amount_vals)):
        """
        this loop saves 'equality' (see function: bbcurve_comparison) for each
        stellar temp in a physically reasonable range in an array
        """
        current_temperature = starttemp + n * jump
        equality_list = np.append(equality_list, bbcurve_comparison(current_temperature,
                                                                    BlueF, GreenF, RedF))
        if equality_list[n] > equality_list[n-1]:
            break # this is to reduce computation time
            # when we get as close to the minimum equality of 3 as possible 
            #  and then start increasing again, we break
        
        
    star_temp = starttemp + jump * (n-1)
    return (star_temp)





"""
here, we calcualte temps for all 70000 stars and add it to the data frame
"""
allStarsDf["Temp"] = " "

#for allStarsDf in tqdm(allStarsDf):
for n in tqdm(range(0, len(allStarsDf))):
    allStarsDf.iloc[n,9] = find_star_temp(allStarsDf.iloc[n,3], allStarsDf.iloc[n,4], allStarsDf.iloc[n,5])
    # col 9 is temp
    
    
# make a histrogram of star temps
number_of_bins = 40
plt.hist(allStarsDf.Temp, number_of_bins)
plt.title("Histogram of All Stellar Temperatures")
plt.xlabel("Stellar Temperature (K)")
plt.ylabel("Occurrence (unitless)")
plt.show()
    

# save temperature data in csv
allStarsDf.to_csv('C:/Users/dylan/Documents/GitHub/DL-Proj/Dylan/Main Files/All_Star_Data_with_Temps.csv')


# now we try find stellar radii for stars we know distances to
goodStarsTemps = allStarsDf.merge(goodStars, 
                                  left_on=['Name', 'X', 'Y', 'BlueF', 'GreenF', 'RedF', 'Parallax',
        'RadialVelocity', 'Variable?' ], 
                                  right_on=['Name', 'X', 'Y', 'BlueF', 'GreenF', 'RedF', 'Parallax',
        'RadialVelocity', 'Variable?'], how='right')

goodStarsTemps["Total_Flux"] = " "

wavelengths = np.linspace(100*10**-9, 10000*10**-9, 99)
for n in range(0,len(goodStarsTemps)):
    temperature = goodStarsTemps.iloc[n, 9] #temps is index 9
    y_integration = bbcurve(wavelengths, temperature) # this is flux/m^2 at surface
    y_integration = y_integration * (goodStarsTemps.iloc[n,3] / bbcurve(400*10**-9, temperature))
  
    #y_integration = y_integration/(goodStarsTemps.iloc[n,14]*3.086*10**16)**2 # this is flux/m^2 at surface

  
    # here we normalise the bbcurve to the given flux data so we can find flux
    # at our satellite
    goodStarsTemps.iloc[n, 16] = integrate.cumtrapz(y_integration, wavelengths)[-1]
    # we use a numerical integration
    # total flux is index 16

    
# we can now calcualte stellar luminosity using L=4 pi r^2 f (use m for dist)
goodStarsTemps["Luminosity"] = 4 * np.pi * (goodStarsTemps.dist*3.086*10**16)**2 * goodStarsTemps.Total_Flux

# we now find stellar radii using emissiivity = 1 for bblack body
goodStarsTemps["Radius"] = (goodStarsTemps.Luminosity / (stef_boltz * (goodStarsTemps.Temp)**4 * 4 * np.pi))**(1/2)

plt.scatter(goodStarsTemps.dist, goodStarsTemps.Radius)

