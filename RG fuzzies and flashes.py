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
plotHomogeneity = True # whether to plot scatter for homogeneity demonstration
plotIsotropy = True

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
        km = KMeans(n_clusters = ngalaxies, random_state = 0)
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

'''     ALTERNATIVE DISTANCE CALIBRATION METHOD  - VARIABLE STARS            '''

var1 = pd.read_csv(r'C:\Users\rgray\OneDrive\ryan\Uni\2023 sem 1\PHYS3080\Assignments\DL-Proj\Dylan\Main Files\VariableClass1.csv')
var2 = pd.read_csv(r'C:\Users\rgray\OneDrive\ryan\Uni\2023 sem 1\PHYS3080\Assignments\DL-Proj\Dylan\Main Files\VariableClass2.csv')
ALLVARS = pd.concat([var1,var2]) # all variable stars

ALLVARS["d_f1"] = (ALLVARS.X - nf1.X)**2 + (ALLVARS.Y - nf1.Y)**2 # distance to flash 1
ALLVARS["d_f2"] = (ALLVARS.X - nf2.X)**2 + (ALLVARS.Y - nf2.Y)**2

top10F1 = ALLVARS.Dist.to_numpy()[ np.argsort(ALLVARS.d_f1.to_numpy()) ][0:10] # 10 closest variable stars
top10F2 = ALLVARS.Dist.to_numpy()[ np.argsort(ALLVARS.d_f2.to_numpy()) ][0:10]

# distance and error as mean and std of 10 closest variable stars
d1_alt = np.mean(top10F1)
d2_alt = np.mean(top10F2)
de_d1_alt = np.std(top10F1)
de_d2_alt = np.std(top10F2)


d1,d2,de_d1,de_d2 = d1_alt,d2_alt,de_d1_alt,de_d2_alt


''' ABSOLUTE PHOTON COUNTS '''

# absolute photon counts, scaled by distance using inverse-square law
absPC1 = nf1["Photon-Count"]*(d1)**2 
de_absPC1 = absPC1 * np.sqrt( 1/np.sqrt(nf1["Photon-Count"]) + \
                             (2*de_d1/d1)**2 ) # uncertainty prop
absPC2 = nf2["Photon-Count"]*(d2)**2
de_absPC2 = absPC2 * np.sqrt( 1/np.sqrt(nf2["Photon-Count"]) + \
                             (2*de_d2/d2)**2 ) # uncertainty prop

absPC = np.mean([absPC1, absPC2]) # absolute magnitude at 1 parsec
de_absPC = np.std([absPC1,absPC2]) # uncertainty

# drop the 2 nearby flashes from the dataframe since they were used for calibration
flashes = flashes.drop(35); flashes = flashes.drop(46)

# give every flash a distance using inverse square law
flashes["Distance"] = np.sqrt(absPC/flashes["Photon-Count"])
flashes["de_Distance"] = flashes.Distance * np.sqrt( 4/flashes["Photon-Count"] + \
                                                    (de_absPC/absPC)**2 ) # unc


'''     BEGIN FUZZY CALIBRATION SECTION             '''

flashes["RV"] = [0 for i,f in flashes.iterrows()] # radial velocity in km/s
flashes["de_RV"] = [0 for i,f in flashes.iterrows()]
flashes["use"] = [0 for i,f in flashes.iterrows()] 

# method below gives each flash the RV of its closest fuzzy
for i,flash in flashes.iterrows():
    camera = flash.Direction
    fuzzies = pd.read_csv(f'DATA//{camera}/Distant_Galaxy_Data.csv')
    
    # find closest fuzzy to each flash
    closestFuzzy = get_obj(flash.X,flash.Y,fuzzies)
    closeFuz = fuzzies.loc[closestFuzzy]
    
    # calculate whether flash is plausibly inside fuzzy galaxy
    sizeD = closeFuz.Size * 1.2 /3600 # maximum size of fuzzy in degrees
    posD = 0.05 # maximum error distance in degrees
    totD = sizeD + posD # maximum allowable distance between fuzzy and flash centers
    withinX = (flash.X - totD <= closeFuz.X <= flash.X + totD)
    withinY = (flash.Y - totD <= closeFuz.Y <= flash.Y + totD)

    flashes.loc[i, 'use'] = (withinX and withinY)
    flashes.loc[i, "RV"] = closeFuz.RadialVelocity
    flashes.loc[i, "de_RV"] = de_fuzzy_RV
flashes = flashes[flashes.use] # only use the flashes localised to a fuzzy

    
'''     BEGIN HUBBLES CONSTANT REGRESSION CALCULATION       '''
        
flashes["Distance_Mpc"] = flashes.Distance / 10**6
flashes["de_Distance_Mpc"] = flashes.de_Distance / 10**6

X = flashes.Distance_Mpc
de_X = flashes.de_Distance_Mpc
Y = flashes.RV
de_Y = flashes.de_RV

def LOBF(X, de_X, Y, de_Y, n_trials):
    ''' Returns m, de_m, c, de_c, using monte-carlo estimation of line of best
    fit from x and y uncertainties, doing n trials. Also returns lines, list of
    lines of best fit from every single trial (to plot density curve)'''
    M = np.zeros((n_trials,))
    C = np.zeros((n_trials,))
    lines = np.zeros((n_trials,2))
    J = len(X) # no of points
    
    for i in range(n_trials):
        # noisy X and noisy Y
        x_trial = X + np.random.randn(J) * de_X
        y_trial = Y + np.random.randn(J) * de_Y
        
        # line of best fit of noisy data
        lobf_trial = np.polyfit(x_trial,y_trial,1) 
        m_trial = lobf_trial[0]; c_trial = lobf_trial[1]
        
        # saving lines
        lines[i] = lobf_trial
        M[i] = m_trial; C[i] = c_trial
        
    # mean of trials as gradient and uncertainty
    m = np.mean(M); c = np.mean(C)
    de_m = np.std(M); de_c = np.std(C)
    
    return m,de_m,c,de_c,lines
    
n_trials = 1000 # number of monte carlo trials
m,de_m,c,de_c,lines = LOBF( X, de_X, Y,de_Y, n_trials ) # line of best fit for hubbles
H0 = m; de_H0 = de_m
print(f"Hubble's constant is {round(H0,2)} +- {round(de_H0,2)} km/s/Mpc")
print(f"Y-intercept is {round(c,2)} +- {round(de_c,2)} km/s")

t_h_s = 1/H0 * 3.086e19
t_h = t_h_s / 3.154e7 / 10**6
de_t_h = t_h * (de_H0/H0)
print(f"Hubble time is {round(t_h_s,2)} seconds and so {round(t_h,2)} +- {round(de_t_h,2)} million years")


if plotHubble:
    # plot data points
    plt.scatter(X, Y, label = "X-ray flashes") 
    plt.errorbar(X,Y, linestyle = 'None', yerr = de_Y, xerr = de_X, )
    
    # plot line of best fit
    predY = np.polyval( [m,c], X )
    plt.plot( X, predY, color = 'black', linewidth = 2, label = "Mean best-fit")
    
    # plot trial lines of best fit
    for line in lines:
        trial_predY = np.polyval(line,X)
        if np.all(line == lines[0]):
            plt.plot(X, trial_predY, 'r-', alpha = .3, label = "Sample best-fits")
        else:
            plt.plot(X, trial_predY, 'r-', alpha = 1/255)
        
        
    # other stuff
    plt.xlabel("Distance (Mpc)"); plt.ylabel("Radial velocity (km/s)")
    plt.title("Distant galaxy movement to determine Hubble's constant")
    plt.legend()
    plt.show()



'''         ASSIGNING DISTANCE TO ALL FUZZIES           '''

all_fuzzies = pd.concat( pd.read_csv(f'DATA//{camera}/Distant_Galaxy_Data.csv') for camera in CAMERAS ) 
# camera from each fuzzy
all_fuzzies["Direction"] = [name.split("DG")[0] for name in all_fuzzies.Name] 

all_fuzzies["Distance"] = all_fuzzies.RadialVelocity / H0
all_fuzzies["de_Distance"] = all_fuzzies.Distance * np.sqrt( (de_fuzzy_RV/all_fuzzies.RadialVelocity)**2 + \
                                                         (de_H0/H0)**2 )
                                                         
maxD = np.max(all_fuzzies.Distance) # max distance Mpc
maxDLY = maxD * 3.26156 # distance in mega light years
print(f"Maximum distance in million light years is {maxDLY}")
                                                         
'''              HOMOGENOUS CALCAULTIONS      '''
                                                         
# max distance of fuzzy
maxR = np.max(all_fuzzies.Distance)

# radius bins, and number of galaxies contained in each bin
R_bins = np.linspace(0, maxR, 11)
n_galaxies = [np.sum(all_fuzzies.Distance<r) for r in R_bins]

# uncertainty: maximum possible number of galaxies in each bin given distance uncertainty
upper_bound_ngalaxy = np.array([ np.sum(all_fuzzies.Distance-all_fuzzies.de_Distance<r) for r in R_bins ]) 
lower_bound_ngalaxy = np.array([ np.sum(all_fuzzies.Distance+all_fuzzies.de_Distance<r) for r in R_bins ])
de_n_galaxies = np.max([ n_galaxies-lower_bound_ngalaxy, upper_bound_ngalaxy-n_galaxies ],axis=0)
de_n_galaxies = de_n_galaxies + (de_n_galaxies==0) # giving each nonzero uncertainty

# remove last point because of faintness of flashes that far away
R_bins = R_bins[:-1]; n_galaxies = n_galaxies[:-1]; de_n_galaxies = de_n_galaxies[:-1]

# get cubic fit
from scipy.optimize import curve_fit
cubic = lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d
popt,pcov = curve_fit( cubic, R_bins, n_galaxies, 
                     sigma = de_n_galaxies, absolute_sigma = True)
print(f"Cubic fit a,b,c,d = \n{[round(i,2) for i in popt]} and uncertainties \
      \n{[round(i,2) for i in np.sqrt(np.diag(pcov))]} ")

if plotHomogeneity:
    # plot scatter points
    plt.scatter(R_bins,n_galaxies, label = "Measured no. of galaxies")
    plt.errorbar( R_bins, n_galaxies, yerr=de_n_galaxies , linestyle = 'None')
    
    # plot fit
    Xfine = np.linspace(0,max(R_bins),1000)
    predY = np.polyval(popt,Xfine)
    plt.plot( Xfine, predY, label = "Cubic fit" )
    
    # other stuff
    plt.legend()
    plt.xlabel("Distance from NE galaxy (Mpc)"); 
    plt.ylabel("Number of galaxies within distance")
    plt.title("Investigating homogeneity of universe")
    plt.show()
    

'''                     ISOTROPIC CALCULATIONS              '''
# number of galaxies in each radius bin, for each camera
ngalcam = {camera:[] for camera in CAMERAS}
de_ngalcam = {camera:[] for camera in CAMERAS}

R_bins_isot = np.linspace( 0, maxR, 11) # less bins since less data in each direction
for camera in CAMERAS:
    # selecting relevant fuzzies
    cam_fuzzies = all_fuzzies[all_fuzzies.Direction == camera]
    
    # number of galaxies contained within each radius
    ngalcam[camera] = [np.sum(cam_fuzzies.Distance<r) for r in R_bins_isot]

    upper_bound_ngalaxy = np.array([ np.sum(cam_fuzzies.Distance-cam_fuzzies.de_Distance<r) for r in R_bins_isot ]) 
    lower_bound_ngalaxy = np.array([ np.sum(cam_fuzzies.Distance+cam_fuzzies.de_Distance<r) for r in R_bins_isot ])
    de_ngalcam[camera] = np.max([ ngalcam[camera]-lower_bound_ngalaxy, upper_bound_ngalaxy-ngalcam[camera] ],axis=0)
    de_ngalcam[camera] = de_ngalcam[camera] + (de_ngalcam[camera]==0) # giving each nonzero uncertainty
    
    # remove last point because of faintness of flashes that far away
    ngalcam[camera] = ngalcam[camera][:-1]
    de_ngalcam[camera] = de_ngalcam[camera][:-1]

if plotIsotropy:
    R_bins_isot = R_bins_isot[:-1] # remove last point
    for camera in CAMERAS:
        plt.scatter(R_bins_isot, ngalcam[camera], label = camera)
        plt.errorbar(R_bins_isot, ngalcam[camera], yerr = de_ngalcam[camera], 
                     linestyle = 'None')
        
    plt.xlabel("Distance from NE galaxy (Mpc)"); 
    plt.ylabel("Number of galaxies within distance")
    plt.title("Investigating isotropy of universe")
    plt.legend()
    plt.show()












    