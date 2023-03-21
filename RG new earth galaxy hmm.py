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
PLANE_CAMERAS = ["Left", "Right", "Front", "Back"]
# theta offsets for stars in the plane
PLANE_OFFSETS = {"Front":0, "Right":90, "Back":180, "Left":270}


'''             STAR DISTANCE CALIBRATION SECTION       '''
parallaxCutoff = 0.01 # minimum parallax to be a valid star for distance calibration

allStarsDf = pd.concat( pd.read_csv(f'DATA//{camera}/Star_Data.csv') for camera in CAMERAS ) # super dataframe containing all stars
goodStars = allStarsDf[allStarsDf.Parallax > parallaxCutoff] # calibrate with star if sufficient parallax
goodStars["Direction"] = [name.split("S")[0] for name in goodStars.Name]

goodStars["dist"] = 1/goodStars.Parallax # distance = 1/parallax where parallax in arcseconds and d in parsec
goodStars["de_dist"] = goodStars.dist * de_parallax / goodStars.Parallax # uncertainty propagation
    
planeStars = goodStars[ goodStars.Direction.isin(PLANE_CAMERAS) ]    
planeStars["theta"] = planeStars.dist + np.array([ PLANE_OFFSETS[camera] for camera in planeStars.Direction ])
planeStars["x_3d"] = planeStars.dist * np.sin(planeStars.theta/360 * 2*np.pi)
planeStars["y_3d"] = planeStars.dist * np.cos(planeStars.theta/360 * 2*np.pi)
planeStars["z_3d"] = planeStars.dist * np.sin(planeStars.Y/360 * 2*np.pi)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter( planeStars.x_3d, planeStars.y_3d, planeStars.z_3d )
plt.show()




