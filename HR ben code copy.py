import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt

import os 

import pandas as pd # pandas is a popular library in industry for manipulating large data tables

# configure notebook for plotting
# %matplotlib inline 
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme

# define default plot settings
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
matplotlib.rcParams['savefig.dpi']= 300             #72 

import warnings
warnings.filterwarnings('ignore')

ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\' 

stars = pd.read_csv(ddir+'Top\\Star_Data.csv') 
print(stars.keys()) # this tells us what column names we have

plt.scatter(stars.X,stars.Y, s = 0.1)
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
# plt.xlim([2,8])
# plt.ylim([-34,-25])
plt.show() # comment

centre = (37,41)
plt.scatter(stars.X,stars.Y)
plt.scatter(*centre,color='C2',marker='X') # * expands the elements of a list 
plt.xlabel('x (pix)')
plt.ylabel('y (pix)');

d = np.sqrt((stars.X-centre[0])** 2 + (stars.Y - centre[1])**2)
galaxy = stars[d<3] # filter to only close ones
plt.scatter(galaxy.X,galaxy.Y,c=galaxy.RadialVelocity,cmap=mpl.cm.seismic) # let's overplot the radial velocities
plt.colorbar()
plt.scatter(*centre,color='C2',marker='X') # * expands the elements of a list 
plt.xlabel('x (pix)')
plt.ylabel('y (pix)');
plt.show()


m0, m1, m2 = (np.log10(galaxy['BlueF']), 
              np.log10(galaxy['GreenF']), 
              np.log10(galaxy['RedF'])) 
colour = m2-m0




s = plt.scatter(colour,m1)
plt.ylabel('Log Flux 1')
plt.xlabel('Log Flux 2 - Log Flux 0')
plt.show()


print('Parallaxes: mean %.3f, sd %.3f' % (np.mean(galaxy['Parallax']),np.std(galaxy['Parallax'])))