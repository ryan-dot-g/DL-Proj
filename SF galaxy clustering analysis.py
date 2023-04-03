# module imports
import numpy as np 
import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os 
import pandas as pd 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
ddir = '/Users/MYMTeam/Desktop/DL-Proj/DATA/' #
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


camera = "Top"; cameraData = TOP
plotPartitions = False # whether to plot clustering scatters
plotHRs = False # whether to plot HR diagrams of each galaxy
plotStarCalibration = False # whether to plot master HR diagram for parallax stars
plotHRcalibration = False # whether to plot fit of HR diagram against star calibration
plotRadVel = False # whether to plot radial velocity graphs of each galaxy
plotRotCurve = True
'''                 BEGIN ACTUAL CLUSTERING SECTION            '''

stars = pd.read_csv(ddir + camera + '/Star_Data.csv') 

if plotPartitions:
    # plot all the stars
    plt.scatter(stars.X,stars.Y, s = 0.1)
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')
    plt.title(f'Star plot ({camera.lower()} camera)')
    plt.grid()
    plt.show() 

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
    km = KMeans(n_clusters = ngalaxies, random_state = 0)
    km.fit(R)
    
    # update master dataframe with the new galaxy
    starPartition["Galaxy"] = gno + km.labels_
    gno += np.max(km.labels_) + 1
    stars.loc[partitionMask, "Galaxy"] = starPartition.Galaxy
    
    if plotPartitions:
        cmap = plt.cm.get_cmap('tab20')
        plt.scatter( starPartition.X, starPartition.Y, c = starPartition.Galaxy, s = 5,
                     cmap = cmap, alpha = 0.5)
        plt.title(f"{camera} camera, partition {i}")
        plt.xlabel("x (pix)"); plt.ylabel("y (pix)")
        plt.axis('equal')
        plt.grid()
        plt.show()
    
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

if plotStarCalibration:
    plt.scatter( goodStars.colour, goodStars.abs_mag, 
                 color = 'C1')
    plt.xlabel('Colour')
    plt.ylabel('Absolute magnitude')
    plt.title("Absolute-magnitude calibrated stars")
    plt.show()
    
'''
DYLAN. The dataframe goodStars (above) has a bunch of stars and distances to each star,
which was calibrated using parallax. Out of all the variable stars, some will be
in the goodStars dataframe. You KNOW the distance for the variable good stars. 
Then you can use this to calibrate and get intrinsic luminosity from the variable good stars
to create a period-intrinsic luminosity relationship. Then you can fit the rest
of the variable good stars to the model to get distances to all the stars.
'''

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
gxy_distances = [] # distance of each galaxy
gxy_bounds  = [] # lower and upper bounds of distance, from uncertainty

for gxy_index in galaxies:
    if gxy_index not in invalidGxyIndices[camera]:
        galaxy = stars[stars.Galaxy == gxy_index]
        
        # get H-R diagram of galaxy
        galaxy.m0, galaxy.m1, galaxy.m2 = (np.log10(galaxy.BlueF), 
                                           np.log10(galaxy.GreenF), 
                                           np.log10(galaxy.RedF) ) 
        galaxy.colour = galaxy.m2 - galaxy.m0
        
        if plotHRs:
            # plot uncalibrated HR diagram
            plt.scatter(galaxy.colour,galaxy.m1)
            plt.ylabel('Log Flux 1')
            plt.xlabel('Log Flux 2 - Log Flux 0')
            plt.title(f"HR diagram of galaxy {gxy_index}, {camera.lower()} camera")
            plt.show()
        
        # offset by difference in max
        dm = np.max(goodStars.abs_mag) - np.max(galaxy.m1)
        delta_dm = 0.1 # uncertainty (eye test) # NEED TO FINISH
        
        galaxyDist = np.power(10, dm/2)
        gxy_distances.append(galaxyDist)
        galaxyBounds = np.power( 10, np.array([dm-delta_dm, dm+delta_dm])/2 )
        gxy_bounds.append(galaxyBounds)
    
        if plotHRcalibration:
            plt.scatter( goodStars.colour, goodStars.abs_mag,
                         color = 'C1') # calibration stars
            plt.scatter( galaxy.colour, galaxy.m1 + dm ) # galaxy
            plt.ylabel('Log Flux 1')
            plt.xlabel('Log Flux 2 - Log Flux 0')
            plt.title(f"HR calibration of galaxy {gxy_index}, {camera.lower()} camera")
            plt.legend(["Benchmark", "Cluster"])
            plt.show()

'''                 BEGIN GALAXY ROT CURVE SECTION            '''
gxy_speeds = []
allTrends = []
G = 4.3e-9
M_L = 1
for gxy_index in galaxies:
    if gxy_index not in invalidGxyIndices[camera]:
        
        galaxy = stars[stars.Galaxy == gxy_index]
        galaxy[galaxy.Parallax < 0.01] #
        medianRV = np.median(galaxy.RadialVelocity)
        gxy_speeds.append(medianRV)
        galaxy.netVel = galaxy.RadialVelocity - medianRV
        
        
        if plotRadVel:
            plt.scatter(galaxy.X, galaxy.Y, c = galaxy.netVel, 
                        cmap = mpl.cm.seismic) # let's overplot the radial velocities
            cbar = plt.colorbar()
            cbar.set_label("Radial velocity (km/s)", rotation = 270)
            plt.xlabel("x (pix)"); plt.ylabel("y (pix)"); 
            plt.title(f"Rot curve of galaxy {gxy_index}, {camera.lower()} camera")
            plt.show()
            
            "Sinead Start"
       
        medianRV = np.median(galaxy.RadialVelocity)
        galaxy.netVel = galaxy.RadialVelocity - medianRV
        galaxy_center = ( np.median(galaxy.X), np.median(galaxy.Y) )
        # g2_central_star = g2[2.41275, -26.31] #find centre of mass, currently median x,y
        # g2_center = np.array([g2_central_star.X, g2_central_star.Y])
        # g2_center = [i[0] for i in g2_center]

        " radius to centre calc"
        galaxy_Xdif = galaxy.X - galaxy_center[0]
        galaxy_Ydif = galaxy.Y - galaxy_center[1]
        galaxy_radii = np.sqrt((galaxy_Xdif**2) + (galaxy_Ydif**2))
        
        " expected & actual mass"
        expected_mass = (5 * galaxy_radii * galaxy.netVel**2) / G
        print('Expected mass:', expected_mass, 'Msun')
        
        # Luminosity = sum(galaxy.GreenF) * 4 * np.pi * distance**2; #need the distance to the galaxy 
        # actual_mass = M_L * Luminosity
        # print('Actual mass:', actual_mass, 'Msun')
        
        
        
        
        
        #X = np.linspace(0,np.max(galaxy_radii),10)
        #Y = []
        #d = (X[1]-X[0] )/2
        #n = 10
        
     
        #for x in X:
               #indbool = (x-d < galaxy_radii ) * (x+d > galaxy_radii) 
               #closestars = galaxy_radii[ indbool ]
               #highestVs = np.sort(galaxy.netVel[ indbool ])[-n:]
               #maxV = np.average(highestVs)
                #maxV = np.max(abs(galaxy.netVel[ indbool ]))
            
        #Y.append(maxV)
        
        "sort data"
        # indRadVel = np.column_stack((galaxy_radii, galaxy.netVel))
        # SortedRadVel = data[np.argsort(data[:,0])]
        
        # X = data[:,0]
        # y = data[:,1]
        # d = (X[1]-X[0] )/2
        # n = 10
        
        # for x in X:
        #        indbool = (x-d < galaxy_radii ) * (x+d > galaxy_radii) 
        #        closestars = galaxy_radii[ indbool ]
        #        highestVs = np.sort(galaxy.netVel[ indbool ])[-n:]
        #        maxV = np.average(highestVs)
        #        maxV = np.max(abs(galaxy.netVel[ indbool ]))
        
        "Test Rot Curve"
        indRadVel = []
        for index, velocity in enumerate(galaxy.netVel):
            indRadVel = np.column_stack((galaxy_radii, galaxy.netVel))
            SortedRadii = indRadVel[np.argsort(indRadVel[:,0])]
        trendRads = []; trendVels = []
        for index, value in enumerate(SortedRadii):
            rad, vel = value
            if index == 0:
                trendRads.append(rad)
                trendVels.append(vel)
            elif index < 100:
                if vel > np.mean(galaxy.netVel):
                    pass
                else:
                    if vel > max(trendVels):
                        trendRads.append(rad)
                        trendVels.append(vel)
            else:
                if vel > max(trendVels):
                    trendRads.append(rad)
                    trendVels.append(vel)
    
        " plot a trendline of the rotation curve "
        x = np.arange(min(galaxy_radii), max(galaxy_radii), 0.01)      
        z,cov = np.polyfit(np.log10(trendRads), trendVels, 1, cov=True)     #logarithmic fit for the data
        p = np.poly1d(z)
        gradUnc, intUnc = np.sqrt(np.diag(cov))
        upper = (z[0] + gradUnc) * np.log10(x) + (z[1] - intUnc)
        lower = (z[0] - gradUnc) * np.log10(x) + (z[1] + intUnc)
        plt.scatter(galaxy_radii, abs(galaxy.netVel))
        plt.xlabel('radius')
        plt.ylabel('velocity')
        plt.title(f"Rotation curve of galaxy {gxy_index}, {camera.lower()} camera")
        plt.plot(x, p(np.log10(x)),"r--", linewidth=0.5)
        plt.fill_between(x, lower, upper, color='r', alpha=0.2)
        plt.show()
        
        allTrends.append(p / p(max(x)))
    
        r2score = r2_score(trendVels, p(np.log10(trendRads)))
    
    
x = np.arange(0.01, 1.01, 0.01)
fig, ax = plt.subplots()
for trendline in allTrends:
    plt.plot(x, trendline(np.log10(x)), color='g', alpha=0.4, linewidth=0.5)
ax.set_ylabel("Rotational Velocity (prop. of v$_{max}$)")
ax.set_xlabel("Radius (prop. of Galactic Radius)")
plt.ylim([0, 0.8])


"end test"

