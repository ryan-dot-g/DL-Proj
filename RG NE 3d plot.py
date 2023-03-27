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
def cube_to_equirect(direction, u, v):
    # convert range -45 to 45 to -1 to 1
    uc = u / 45
    vc = v / 45
    if direction == "Front": # POSITIVE X
        x = 1
        y = vc
        z = -uc 
    elif direction == "Back":  # NEGATIVE X
        x = -1
        y = vc
        z = uc
    elif direction == "Top": # POSITIVE Y
        x = uc
        y = 1
        z = -vc
    elif direction == "Bottom": # NEGATIVE Y
        x = uc
        y = -1
        z = vc
    elif direction == "Left": # POSITIVE Z
        x = uc
        y = vc
        z = 1
    else: # direction == "Right": # NEGATIVE Z
        x = -uc
        y = vc
        z = -1 
    # now to convert the XYZ to spherical coordinates
    # this is using the physics convention of spherical coords!
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(z, x)
    theta = np.arccos(y / r)

    theta = theta * 180 / np.pi
    azimuth = (- azimuth + np.pi) * 360 / (2 * np.pi)
    
    return azimuth, theta

for i, camera in enumerate(CAMERAS):
        # read the data from the .txt file into a dataframe
        stardata = pd.read_csv(ddir + f'{camera}\\Star_Data.csv')  
        stardata = stardata[stardata.Parallax > 0.01] # only our galaxy stars
        u = stardata["X"].to_numpy(); v = stardata["Y"].to_numpy() # convert X and Y data to "U" and "V" data
        azimuth, theta = cube_to_equirect(camera, u, v) # perform the coordinate transform
        azimuth = np.around(azimuth, decimals=4); theta = np.around(theta, decimals=4) # round to appropriate decimals
        
        # now overwrite the old coordinates with the new ones
        stardata["Equat"] = azimuth
        stardata["Polar"] = theta
        
        if i == 0:
            # if this is the first iteration, write to a new DataFrame that will store all of the star data
            all_stardata = stardata
        else:
            all_stardata = pd.concat([all_stardata, stardata]) # add this face stardata to the rest of the data


all_stardata["dist"]= 1/all_stardata.Parallax
asd = all_stardata # shorthand
theta = np.deg2rad(asd.Equat); phi = np.deg2rad( asd.Polar ) # shorthand
asd["xplot"] = asd.dist * np.sin(phi) * np.cos(theta)
asd["yplot"] = asd.dist * np.sin(phi) * np.sin(theta)
asd["zplot"] = asd.dist * np.cos(phi)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter( asd.xplot, asd.yplot, asd.zplot, s = 0.3, label = "Star")
ax.set(xlabel = "X (pc)", ylabel = "Y (pc)", zlabel = "Z (pc)", title = "New Earth Galaxy")
ax.scatter( [0], [0], [0], s = 10, color = 'red', label = "New Earth")
ax.grid(False)
ax.legend()
ax.set_aspect('equal')
plt.show()

# allStarsDf = pd.concat( pd.read_csv(f'DATA//{camera}/Star_Data.csv') for camera in CAMERAS )
# P = allStarsDf.Parallax[allStarsDf.Parallax>0]
# P = 1/P
# X = np.sort(P)

# plt.hist(X)
# plt.xlabel("1/parallax (pc)")
# plt.ylabel("Number of stars within x range")
# plt.title("High-parallax star density")
# plt.show()