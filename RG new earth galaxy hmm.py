# module imports
import numpy as np 
import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os 
import pandas as pd 
ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\'

# define default plot settings
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
# matplotlib.rcParams['savefig.dpi']= 300             #72 



CAMERAS = ["Top", "Bottom", "Left", "Right", "Front", "Back"] 
de_parallax = 0.001 # parallax uncertainty, in arcseconds
de_flux_pct = 0.01 # percentage uncertainty of flux, in W/nm/m^2
de_offset = np.sqrt( 2* (1/np.log(10) * de_flux_pct)**2 ) # error of offset (magnitude)


'''             STAR DISTANCE CALIBRATION SECTION       '''
parallaxCutoff = 0.01 # minimum parallax to be a valid star for distance calibration

allStarsDf = pd.concat( pd.read_csv(f'DATA//{camera}/Star_Data.csv') for camera in CAMERAS ) # super dataframe containing all stars
goodStars = allStarsDf[allStarsDf.Parallax > parallaxCutoff] # calibrate with star if sufficient parallax
goodStars["m0"], goodStars["m1"], goodStars["m2"] = (np.log10(goodStars.BlueF),
                                                     np.log10(goodStars.GreenF),
                                                     np.log10(goodStars.RedF)) # different color fluxes
goodStars["colour"] = goodStars.m2 - goodStars.m0


goodStars["dist"] = 1/goodStars.Parallax # distance = 1/parallax where parallax in arcseconds and d in parsec
goodStars["de_dist"] = goodStars.dist * de_parallax / goodStars.Parallax # uncertainty propagation

goodStars["abs_mag"] = goodStars.m1 + 2 * np.log10(goodStars.dist) # absolute magnitude using log scale
goodStars["de_abs_mag"] = np.sqrt( ((1/np.log(10))*de_flux_pct)**2 + \
                                  ((2/np.log(10))*goodStars.de_dist/goodStars.dist)**2   ) # uncertainty prop