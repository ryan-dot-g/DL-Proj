# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:43:11 2023

@author: dylan
"""

import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


## stellar temperature fit using colour filters and blackbody curves

## UNCERTAINTIES!!!

# stuff
# constants
h = 6.626 * 10**-34 
c = 3 * 10**8 ;
k = 1.38064 * 10**-23 
#t = 6000

'''
#filter_wv = np.array(440, 500, 700) * 10**-9
star1_monochrom_lum = [6.58E-20,	7.17E-20,	6.23E-20]
wavelengths = np.linspace(0, 2500, 2500) * 10**-9

# eq with temperature input
B = 2*h*c**2 / ( ((wavelengths**5) * (np.exp(h*c/(wavelengths*k*t)) - 1)) )

plt.plot(wavelengths,B)
plt.show()
'''

x = np.array([440,500,700]) * 10**-9
y = [1.06E-17	,8.86E-18	,4.35E-18]
yratio = np.array(y) / y[0]

def bbcurve(wavelength, temperature):
    return (2*h*c**2) / (wavelength**5 * ( np.exp((h*c)/(wavelength*k*temperature)) - 1))

# plan:
# 

# initialize some points
x_data = np.linspace(min(x), max(x), 50)
# transform x_data to y-axis values via poly3d
y_data = bbcurve(x_data, 5000)

plt.plot(x,y, 'ro', x_data, y_data)


