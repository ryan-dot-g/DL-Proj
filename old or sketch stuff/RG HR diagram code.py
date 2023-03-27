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

# set data directory and read data
camera = 'Top'
stars = pd.read_csv(ddir + camera + '\\Star_Data.csv') 
print(stars.keys()) # this tells us what column names we have

# plot all the stars
plt.scatter(stars.X,stars.Y, s = 0.1)
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title(f'Star plot ({camera.lower()} camera)')
plt.show() 

'''             PER GALAXY SECTION          '''
# define a galaxy - eyesight clustering
centre = (-38,34)
radius = 5
d = np.sqrt((stars.X-centre[0])** 2 + (stars.Y - centre[1])**2)
galaxy = stars[d<radius] # filter to only close ones

# plot radial velocities within galaxy
plt.scatter(galaxy.X, galaxy.Y, c = galaxy.RadialVelocity, 
            cmap = mpl.cm.seismic) # let's overplot the radial velocities
cbar = plt.colorbar()
cbar.set_label("Radial velocity (km/s)", rotation = 270)
plt.scatter(*centre,color='C2',marker='X') # * expands the elements of a list 
plt.xlabel('x (pix)')
plt.ylabel('y (pix)');
plt.title("Radial velocity of galaxy")
plt.show()

# plot H-R diagram of galaxy
m0, m1, m2 = (np.log10(galaxy.BlueF), 
              np.log10(galaxy.GreenF), 
              np.log10(galaxy.RedF) ) 
colour = m2-m0
plt.scatter(colour,m1)
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')
plt.title("HR diagram of galaxy")
plt.show()

'''                 END GALAXY SECTION          '''

'''             STAR DISTANCE CALIBRATION SECTION       '''

import glob
all_stars = glob.glob(ddir + '*/Star_Data.csv') # all star data
parallaxCutoff = 0.01 # minimum parallax to be a valid star for distance calibration

allStarsDf = pd.concat( pd.read_csv(catalog) for catalog in all_stars ) # super dataframe containing all stars
goodStars = allStarsDf[allStarsDf.Parallax > parallaxCutoff] # calibrate with star if sufficient parallax
goodStars["m0"], goodStars["m1"], goodStars["m2"] = (np.log10(goodStars.BlueF),
                                                     np.log10(goodStars.GreenF),
                                                     np.log10(goodStars.RedF)) # different color fluxes
goodStars["colour"] = goodStars.m2 - goodStars.m0
goodStars["dist"] = 1/goodStars.Parallax # distance = 1/parallax where parallax in arcseconds
goodStars["abs_mag"] = goodStars.m1 + 2 * np.log10(goodStars.dist) # absolute magnitude using log scale
plt.scatter( goodStars.colour, goodStars.abs_mag, 
             color = 'C1')
plt.xlabel('Colour')
plt.ylabel('Absolute magnitude')
plt.title("Absolute-magnitude calibrated stars")
plt.show()

'''         END STAR DISTANCE CALIBRATION SECTION   '''

'''         GALAXY DISTANCE MATCHING SECTION        '''
# distance modulus to adjust (eye test) 6.8
starsInRange = goodStars[ (min(colour) <= goodStars.colour) *\
                         goodStars.colour <= max(colour) ]
dm = np.mean( starsInRange.abs_mag ) - np.mean( m1 )
dm = 6.2 
delta_dm = 0.1 # uncertainty (eye test)

plt.scatter( goodStars.colour, goodStars.abs_mag, 
             color = 'C1') # calibration stars
plt.scatter( colour, m1 + dm ) # galaxy
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')
plt.title("Cluster calibration")
plt.legend(["Benchmark", "Cluster"])
plt.show()

galaxyDist = np.power( 10, np.array([dm-delta_dm, dm+delta_dm])/2 )
print(galaxyDist)
