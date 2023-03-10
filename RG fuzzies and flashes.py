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

'''                 BEGIN CLUSTERING SECTION            '''

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

# idk fam tbh
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

'''                 BEGIN USER INPUT SECTION            '''

CAMERAS = {"Top":TOP,"Bottom":BOTTOM,"Left":LEFT,
           "Right":RIGHT,"Front":FRONT,"Back":BACK}

STARS = {}

OBJ_DISTANCES = {}
OBJ_SPEEDS = {}

'''                 BEGIN ACTUAL CLUSTERING SECTION            '''

for camera,cameraData in CAMERAS.items():

    stars = pd.read_csv(ddir + camera + '\\Star_Data.csv') 
    
    gno = 0 # how many galaxies already found
    for i in range(len(cameraData[0])):
        xmin,xmax = cameraData[0][i]
        ymin,ymax = cameraData[1][i]
        ngalaxies = cameraData[2][i]
        
        partitionMask = (xmin <= stars.X) * (stars.X < xmax) * \
                        (ymin <= stars.Y) * (stars.Y < ymax) 
        starPartition = stars[ partitionMask ]
        
        # perform clustering fit
        R = np.array([ i for i in zip(*[starPartition.X,starPartition.Y]) ]) # (x,y) coords
        km = KMeans(n_clusters = ngalaxies)
        km.fit(R)
        
        # update master dataframe with the new galaxy
        starPartition["Galaxy"] = gno + km.labels_
        gno += np.max(km.labels_) + 1
        stars.loc[partitionMask, "Galaxy"] = starPartition.Galaxy
        
    stars.Galaxy = stars.Galaxy.astype(int)
    galaxies = set(stars.Galaxy)
    
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
    
    # galaxies that are excluded from analysis because they aren't really a galaxy
    # e.g. 2 or 3 galaxies that are bunched up together, or new earth galaxy stars
    invalidGxyIndices = {"Top": [4,6,8], 
                         "Bottom": [6,8,15],
                         "Left": [0,7,8,10,11,15,18,25], 
                         "Right": [1,3,5,6,7,8,12,13],
                         "Front": [0,6,34,35],
                         "Back": [0,1,3,4,5,6,11,13,14,16]
                         }
    
    '''                 BEGIN GALAXY HR AND DISTANCE SECTION            '''
    gxy_distances = {} # distance of each galaxy
    gxy_bounds  = [] # lower and upper bounds of distance, from uncertainty
    gxy_speeds = {}
    
    for gxy_index in galaxies:
        if gxy_index not in invalidGxyIndices[camera]:
            galaxy = stars[stars.Galaxy == gxy_index]
            
            # get H-R diagram of galaxy
            galaxy.m0, galaxy.m1, galaxy.m2 = (np.log10(galaxy.BlueF), 
                                               np.log10(galaxy.GreenF), 
                                               np.log10(galaxy.RedF) ) 
            galaxy.colour = galaxy.m2 - galaxy.m0
            
            # offset by difference in max
            dm = np.max(goodStars.abs_mag) - np.max(galaxy.m1)
            delta_dm = 0.1 # uncertainty (eye test) # NEED TO FINISH
            
            galaxyDist = np.power(10, dm/2)
            gxy_distances[gxy_index] = galaxyDist

            galaxyBounds = np.power( 10, np.array([dm-delta_dm, dm+delta_dm])/2 )
            gxy_bounds.append(galaxyBounds)
            
            medianRV = np.median(galaxy.RadialVelocity)

            gxy_speeds[gxy_index] = medianRV

    OBJ_DISTANCES[camera] = gxy_distances
    OBJ_SPEEDS[camera] = gxy_speeds

    STARS[camera] = stars

def get_galaxy(X,Y,stars,camera):
    ''' Takes X,Y pixel location of a flash. Returns the number galaxy it belongs
    to in the star dataframe with galaxies from that camera. Returns None if
    (a) the flash is not contained within any 1 galaxy or
    (b) the galaxy it is contained in is invalid in the list of invalid galaxies '''
    
    return 1
    

'''         BEGIN FLASH CALIBRATION SECTION             '''

flashes = pd.read_csv(ddir + 'Flash_Data.csv')
nf1, nf2 = flashes.loc[35], flashes.loc[46] # selecting the 2 super bright flashes

g1 = get_galaxy(nf1.X, nf1.Y, STARS[nf1.Direction], nf1.Direction) 
g2 = get_galaxy(nf2.X, nf2.Y, STARS[nf2.Direction], nf2.Direction) 

d1 = OBJ_DISTANCES[camera][g1]  # distance of near flash 1
d2 = OBJ_DISTANCES[camera][g2] # distance of near flash 2

absPC1 = 0; # absolute photon count, scaling by distance
absPC2 = 0; 