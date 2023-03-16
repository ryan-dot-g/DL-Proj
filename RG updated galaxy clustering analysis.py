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

'''                 BEGIN PRELIM DATA SECTION            '''
from sklearn.cluster import KMeans
    
pmin,pmax = -60, 60 # minimum and maximum pixel number in each image

# teensy bit of bleeding partition 1 on big galaxy
BOTTOM = [
    [ [pmin,20],   [20,pmax]   ],
    [ [pmin,pmax], [pmin,pmax] ],
    [     3,           14      ]
    ]

# bleeding in partition 1
TOP = [
    [ [pmin,00],   [00,10],    [00,10]    ],
    [ [pmin,pmax], [pmin,-20], [-20,pmax] ],
    [    1,            8,          6      ]
    ]

# partition 0 is galactic disk stars
# bleeding in partition 3,4
LEFT = [
    [ [pmin,pmax], [pmin,pmax], [pmin,-20], [-20,20],  [20,25.5], [25.5,pmax] ],
    [ [-30,20],    [pmin,-30],  [20,pmax],  [20,pmax], [20,pmax], [20,pmax]   ],
    [     1,            5,          1,         5,         7,          7       ]
    ]

# partition 1 is galactic disk stars
# bleeding partition 2
RIGHT = [
    [ [pmin,-10], [pmin,-10], [-10,10],    [10,pmax]   ],
    [ [5,pmax],   [pmin,5],   [pmin,pmax], [pmin,pmax] ],
    [    3,          1,           4,            6      ]
    ]

# partition 0 is a mess
# partition 2 is just a random star lol
# partition 4 is galactic disk stars
BACK = [
    [ [pmin,-35], [pmin,-35], [pmin,-35], [-35,pmax], [-35,pmax], [-35,pmax] ],
    [ [pmin,2],   [2,10],     [10,pmax],  [pmin,-10], [-10,20],   [20,pmax]  ],
    [     7,        7,           1,            1,         1,          1      ]
    ]

# quite a few clustering problems
FRONT = [
    [ [pmin,-37],  [-37,-20],  [-37,-20],  [-20,pmax], [-20,pmax] ],
    [ [pmin,pmax], [7.5,pmax], [pmin,7.5], [-30,pmax], [pmin,-30] ],
    [    1,           33,           1,          1,          1     ]
    ]

DEFAULT = [
    [ [pmin,pmax] ],
    [ [pmin,pmax] ],
    [ 10 ]
    ]

# galaxies that are excluded from analysis because they aren't really a galaxy
# e.g. 2 or 3 galaxies that are bunched up together, or new earth galaxy stars
invalidGxyIndices = {"Top": [4,6,8], 
                     "Bottom": [6,8,15],
                     "Left": [0,7,8,10,11,15,18,25], 
                     "Right": [1,3,5,6,7,8,12,13],
                     "Front": [0,6,34,35],
                     "Back": [0,1,3,4,5,6,11,13,14,16]
                     }


'''                 BEGIN RAW UNCERTAINTY SECTION           '''
de_parallax = 0.001 # parallax uncertainty, in arcseconds
de_flux_pct = 0.01 # percentage uncertainty of flux, in W/nm/m^2
de_offset = np.sqrt( 2* (1/np.log(10) * de_flux_pct)**2 ) # error of offset (magnitude)

de_X, de_Y = 0.0001, 0.0001 # uncertainty of X,Y positions in sky, in degrees
de_RV = 0.03 # uncertainty of radial velocity, in km/s

'''                 BEGIN DATA COLLECTION SECTION            '''
CAMERAS = ["Top", "Bottom", "Left", "Right", "Front", "Back"] 

# PLOTTING BOOLS
camera = "Top"; cameraData = TOP
plotPartitions = True # whether to plot clustering scatters
plotHRs = True # whether to plot HR diagrams of each galaxy
plotStarCalibration = False # whether to plot master HR diagram for parallax stars
plotHRcalibration = True # whether to plot fit of HR diagram against star calibration

# dictionary linking each camera to the (xmin,xmax,ymin,ymax,n_galaxies)
# data which partitions the camera's field into sections that can be well
# clustered into galaxies
cam_partition_data = {"Top":TOP,"Bottom":BOTTOM,"Left":LEFT,
                      "Right":RIGHT,"Front":FRONT,"Back":BACK}


# dictionary { camera: dataframe of all stars from that camera }
STARS = {camera:None for camera in CAMERAS}

# dictionaries {camera: dictionary {galaxy index: data about that galaxy} }

# distance from NE to each galaxy
GAL_DISTANCES = {camera:{} for camera in CAMERAS} 

# median radial velocity of all stars in each galaxy. 
# Median excludes dodgy outlier stars not actually in gal
GAL_RV = {camera:{} for camera in CAMERAS}
 
# (x,y) median position of all stars in galaxy, proxy for galactic center
GAL_CENTERS = {camera:{} for camera in CAMERAS} 

# uncertainty dictionaries
DE_GAL_DISTANCES = {camera:{} for camera in CAMERAS}
DE_GAL_RV = {camera:{} for camera in CAMERAS}
DE_GAL_CENTERS = {camera:{} for camera in CAMERAS}

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

'''                 BEGIN ACTUAL CLUSTERING SECTION            '''

for camera in CAMERAS:
    cameraData = cam_partition_data[camera]  

    stars = pd.read_csv(ddir + camera + '\\Star_Data.csv') 
    STARS[camera] = stars
    
    if plotPartitions:
        # plot all the stars
        plt.scatter(stars.X,stars.Y, s = 0.1)
        plt.xlabel('x (degrees)')
        plt.ylabel('y (degrees)')
        plt.title(f'Star plot ({camera.lower()} camera)')
        plt.grid()
        plt.show() 
    
    gno = 0 # how many galaxies already found
    for i in range(len(cameraData[0])):
        # getting partition data for each partition of each camera
        xmin,xmax = cameraData[0][i]
        ymin,ymax = cameraData[1][i]
        ngalaxies = cameraData[2][i]
        
        # bool whether each star is in partition, then select the star partition
        partitionMask = (xmin <= stars.X) * (stars.X < xmax) * \
                        (ymin <= stars.Y) * (stars.Y < ymax) 
        starPartition = stars[ partitionMask ]
        
        # perform clustering fit
        R = np.array([ i for i in zip(*[starPartition.X,starPartition.Y]) ]) # (x,y) coords
        km = KMeans(n_clusters = ngalaxies)
        km.fit(R)
        
        # update master dataframe with the new galaxy
        starPartition["Galaxy"] = gno + km.labels_ # new galaxy number
        gno += np.max(km.labels_) + 1 # update galaxy number
        stars.loc[partitionMask, "Galaxy"] = starPartition.Galaxy
        
        if plotPartitions:
            cmap = plt.cm.get_cmap('tab20')
            plt.scatter( starPartition.X, starPartition.Y, c = starPartition.Galaxy, s = 5,
                         cmap = cmap, alpha = 0.5)
            plt.title(f"{camera} camera galaxy clustering, partition {i}")
            plt.xlabel("x (degrees)"); plt.ylabel("y (degrees)")
            plt.axis('equal')
            plt.grid()
            plt.show()
        
    stars.Galaxy = stars.Galaxy.astype(int)
    galaxies = set(stars.Galaxy)
        
    
    '''                 BEGIN GALAXY HR AND DISTANCE SECTION            ''' 
    for gxy_index in galaxies:
        if gxy_index not in invalidGxyIndices[camera]: # exclude partitions that don't contain a good galaxy
            galaxy = stars[stars.Galaxy == gxy_index] # select galaxy
            
            # get H-R diagram of galaxy
            galaxy.m0, galaxy.m1, galaxy.m2 = (np.log10(galaxy.BlueF), 
                                               np.log10(galaxy.GreenF), 
                                               np.log10(galaxy.RedF) ) 
            galaxy.colour = galaxy.m2 - galaxy.m0
            
            # offset by difference in max
            galacticOffset = np.max(goodStars.abs_mag) - np.max(galaxy.m1)
            de_offset; # uncertainty propagation
            
            if plotHRs:
                # plot uncalibrated HR diagram
                plt.scatter(galaxy.colour,galaxy.m1)
                plt.ylabel('Log Flux 1')
                plt.xlabel('Log Flux 2 - Log Flux 0')
                plt.title(f"HR diagram of galaxy {gxy_index}, {camera.lower()} camera")
                plt.show()
                # uncertainties so small they have been omitted
            
            if plotHRcalibration:
                plt.scatter( goodStars.colour, goodStars.abs_mag, 
                             color = 'C1') # calibration stars
                plt.scatter( galaxy.colour, galaxy.m1 + galacticOffset, color = 'C0' ) # galaxy
                plt.ylabel('Log Flux 1')
                plt.xlabel('Log Flux 2 - Log Flux 0')
                plt.title(f"HR calibration of galaxy {gxy_index}, {camera.lower()} camera")
                plt.legend(["Benchmark", "Calibrated galaxy"])
                plt.show()
                # uncertainties so small they have been omitted
            
            # calculate galaxy distance and uncertainty
            galaxyDist = np.power(10, galacticOffset/2) 
            GAL_DISTANCES[camera][gxy_index] = galaxyDist
            de_galaxyDist = galaxyDist * np.log(10)/2 * de_offset
            DE_GAL_DISTANCES[camera][gxy_index] = de_galaxyDist
            
            # calculate galaxy radial velocity and uncertainty
            medianRV = np.median(galaxy.RadialVelocity)
            GAL_RV[camera][gxy_index] = medianRV
            DE_GAL_RV[camera][gxy_index] = de_RV
            
            # calculate position of galactic center and uncertainty
            GAL_CENTERS[camera][gxy_index] = (np.median(galaxy.X),np.median(galaxy.Y))
            DE_GAL_CENTERS[camera][gxy_index] = (de_X, de_Y)