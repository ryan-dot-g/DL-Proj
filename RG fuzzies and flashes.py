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

de_PC = 174 # uncertainty in photon counts
de_fuzzy_RV = 0.1 # uncertainty in radial velocity

'''                 BEGIN DATA COLLECTION SECTION            '''
CAMERAS = ["Top", "Bottom", "Left", "Right", "Front", "Back"] 

# PLOTTING BOOLS
plotHubble = True # whether to plot scatter for hubbles constant 


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

def get_galaxy(X,Y,stars,camera):
    ''' Takes X,Y pixel location of a flash. Returns the number galaxy it belongs
    to in the star dataframe with galaxies from that camera. 
    
    IDEALLY Returns None if
    (a) the flash is not contained within any 1 galaxy or
    (b) the galaxy it is contained in is invalid in the list of invalid galaxies.
    Currently not implemented'''
    
    gcenters = GAL_CENTERS[camera]
    gdistances = {i: np.linalg.norm((X-r[0],Y-r[1])) for i,r in gcenters.items()}
    
    closest_galaxy = [i for i,d in gdistances.items() if d==min(dist for dist in gdistances.values())][0]
    
    return closest_galaxy

def get_obj(X,Y,objs):
    ''' Takes X,Y pixel location of an event observation. Returns the number
    object it belongs to in the objs dataframe with objs from that camera '''    
    obj_distances = {i: np.linalg.norm((X-f.X,Y-f.Y)) for i,f in objs.iterrows()}
    closest_obj = [i for i,d in obj_distances.items() if d==min(dist for dist in obj_distances.values())][0]
    
    return closest_obj
    

'''         BEGIN FLASH CALIBRATION SECTION             '''

flashes = pd.read_csv(ddir + 'Flash_Data.csv')
nf1, nf2 = flashes.loc[35], flashes.loc[46] # selecting the 2 super bright flashes

g1 = get_galaxy(nf1.X, nf1.Y, STARS[nf1.Direction], nf1.Direction) 
g2 = get_galaxy(nf2.X, nf2.Y, STARS[nf2.Direction], nf2.Direction) 

d1 = GAL_DISTANCES[nf1.Direction][g1]  # distance of near flash 1
de_d1 = DE_GAL_DISTANCES[nf1.Direction][g1] # uncertainty
d2 = GAL_DISTANCES[nf2.Direction][g2] # distance of near flash 2
de_d2 = DE_GAL_DISTANCES[nf2.Direction][g2] 

# absolute photon counts, scaled by distance using inverse-square law
absPC1 = nf1["Photon-Count"]*(d1)**2 
de_absPC1 = absPC1 * np.sqrt( (de_PC/nf1["Photon-Count"])**2 + \
                             (2*de_d1/d1)**2 ) # uncertainty prop
absPC2 = nf2["Photon-Count"]*(d2)**2
de_absPC2 = absPC2 * np.sqrt( (de_PC/nf2["Photon-Count"])**2 + \
                             (2*de_d2/d2)**2 ) # uncertainty prop

absPC = np.mean([absPC1, absPC2]) # absolute magnitude at 1 parsec
de_absPC = np.std([absPC1,absPC2]) # uncertainty

# drop the 2 nearby flashes from the dataframe since they were used for calibration
flashes = flashes.drop(35); flashes = flashes.drop(46)

# give every flash a distance using inverse square law
flashes["Distance"] = np.sqrt(absPC/flashes["Photon-Count"])
flashes["de_Distance"] = flashes.Distance * np.sqrt( (2*de_PC/flashes["Photon-Count"])**2 + \
                                                    (de_absPC/absPC)**2 ) # unc


'''     BEGIN FUZZY CALIBRATION SECTION             '''
FUZ_DISTANCES = {cam:{} for cam in CAMERAS}
DE_FUZ_DISTANCES = {cam:{} for cam in CAMERAS}
FUZ_RADVEL = {cam:{} for cam in CAMERAS}
DE_FUZ_RADVEL = {cam:{} for cam in CAMERAS}

# method below gives every fuzzy a distance by assigning nearest flash
# for camera in CAMERAS.keys():
#     fuzzies = pd.read_csv(f'DATA//{camera}/Distant_Galaxy_Data.csv')
#     FUZZIES[camera] = fuzzies
    
#     FUZ_DISTANCES[camera] = {}
#     FUZ_RADVEL[camera] = {}
#     for i,fuzzy in fuzzies.iterrows():
#         # locate closest flash and get its distance
#         closestFlash = get_obj(fuzzy.X,fuzzy.Y,flashes,camera) # need to fiz and index flaashes with camera first
#         distance = flashes.loc[closestFlash].Distance
        
#         # assign fuzzy galaxy that distance, record rad vel
#         FUZ_DISTANCES[camera][i] = distance
#         FUZ_RADVEL[camera][i] = fuzzy.RadialVelocity

# method below gives only the closest fuzzy to each flash a distance

for i,flash in flashes.iterrows():
    camera = flash.Direction
    fuzzies = pd.read_csv(f'DATA//{camera}/Distant_Galaxy_Data.csv')
    
    # find closest fuzzy to each flash
    closestFuzzy = get_obj(flash.X,flash.Y,fuzzies)
    
    # assign distance and radial velocity of the fuzzy
    FUZ_DISTANCES[camera][closestFuzzy] = flash.Distance
    DE_FUZ_DISTANCES[camera][closestFuzzy] = flash.de_Distance
    
    FUZ_RADVEL[camera][closestFuzzy] = fuzzies.loc[closestFuzzy].RadialVelocity
    DE_FUZ_RADVEL[camera][closestFuzzy] = de_fuzzy_RV 
    
        
# flash distances and radial velocities 
flash_distance_parsec = np.array([d for camDistances in FUZ_DISTANCES.values() for d in camDistances.values()])
de_flash_distance_parsec = np.array([de_d for de_camDistances in DE_FUZ_DISTANCES.values() for de_d in de_camDistances.values()])

flash_RV = np.array([rv for camRV in FUZ_RADVEL.values() for rv in camRV.values()])
de_flash_RV = np.array([de_rv for de_camRV in DE_FUZ_RADVEL.values() for de_rv in de_camRV.values()])


# exclude galaxies in local cluster
nonlocal_cluster_flashes = [True for i in flash_RV] # (flash_distance_parsec>0.05*10**6)
flash_distance_parsec = flash_distance_parsec[nonlocal_cluster_flashes]
de_flash_distance_parsec = de_flash_distance_parsec[nonlocal_cluster_flashes]
flash_RV = flash_RV[nonlocal_cluster_flashes]
de_flash_RV = de_flash_RV[nonlocal_cluster_flashes]

# distances, uncertainties in megaparsecs for Hubbles constant units
flash_distance_Mpc = flash_distance_parsec/10**6 
de_flash_distance_Mpc = de_flash_distance_parsec/10**6

bestFit = np.polyfit(flash_distance_Mpc,flash_RV,1) # line of best fit
H0 = bestFit[0]
print(f"Hubble's constant is {H0}")
predRV = np.polyval(bestFit,flash_distance_Mpc)

if plotHubble:
    plt.scatter(flash_distance_Mpc, flash_RV)
    plt.plot(flash_distance_Mpc, predRV, '--')
    plt.xlabel("Distance (Mpc)"); plt.ylabel("Radial velocity (km/s)")
    plt.title("Distant galaxy movement to determine Hubble's constant")
    plt.show()



