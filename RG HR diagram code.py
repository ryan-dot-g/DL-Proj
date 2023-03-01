# module imports
import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os 
import pandas as pd # pandas is a popular library in industry for manipulating large data tables


# define default plot settings
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
matplotlib.rcParams['savefig.dpi']= 300             #72 

# set data directory and read data
ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\' 
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
centre = (37,41)
radius = 3
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
s = plt.scatter(colour,m1)
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')
plt.title("HR diagram of galaxy")
plt.show()

# output galaxy parallaxes
print('Parallaxes (arcsec): mean %.3f, sd %.3f' % (np.mean(galaxy.Parallax),np.std(galaxy.Parallax)))

'''                 END GALAXY SECTION          '''

'''             START DISTANCE ESTIMATION SECTION       '''

import glob
all_stars = glob.glob(ddir + '*/Star_Data.csv') # all star data

for i, catalog in enumerate(all_stars):
    this = pd.read_csv(catalog)
    thispar = this.Parallax
    thism0, thism1, thism2 = (np.log10(this.BlueF),
                              np.log10(this.GreenF),
                              np.log10(this.RedF))
    thiscolour = thism2 - thism0
    dist = 1/thispar
    abs_mag = thism1 + 2 * np.log10(dist)
    mm = thispar > 0.01 
    plt.scatter(thiscolour[mm], abs_mag[mm], color = 'C1')

plt.xlabel('Log Flux 2 - Log Flux 0')
plt.ylabel('Log Flux 1')
plt.title("Stars with high parallax")
plt.show()
        