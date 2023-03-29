# module imports
import numpy as np 
import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os 
import pandas as pd 
ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\'

# ddir dylan
# ddir = 'C:/Users/dylan/Documents/GitHub/DL-Proj/DATA/'

# define default plot settings
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
# matplotlib.rcParams['savefig.dpi']= 300             #72 

''' BEGIN SECTION                   '''

CAMERAS = ["Top", "Bottom", "Left", "Right", "Front", "Back"] 
de_parallax = 0.001 # parallax uncertainty, in arcseconds
de_flux_pct = 0.01 # percentage uncertainty of flux, in W/nm/m^2

parallaxCutoff = 0.01 # minimum parallax to be a valid star for distance calibration

allStarsDf = pd.concat( pd.read_csv(f'DATA//{camera}/Star_Data.csv') for camera in CAMERAS ) # super dataframe containing all stars


allStarsDf = pd.read_csv(r'C:\Users\rgray\OneDrive\ryan\Uni\2023 sem 1\PHYS3080\Assignments\DL-Proj\Dylan\Main Files\All_Star_Data_with_Temps.csv')
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
    
maxt, mint = max(goodStars.Temp), min(goodStars.Temp)
tempsizemap = lambda t: (t-mint)/(maxt-mint) * 15 + 5

goodStars['tempsize'] = tempsizemap(goodStars.Temp)
goodStars["Plotcol"] = [ 255*np.array([g.m0,g.m1,g.m2])/(g.m0+g.m1+g.m2) for i,g in goodStars.iterrows()]
    
varbYes = goodStars[goodStars["Variable?"]==1]
varbNo = goodStars[goodStars["Variable?"]==0]

plt.scatter(varbNo.colour, varbNo.abs_mag,
            marker = '.', c = varbNo.Temp, cmap = mpl.cm.seismic, label = "Non-variable stars")
cbar = plt.colorbar()
cbar.set_label("Temperature (K)")
plt.scatter(varbYes.colour, varbYes.abs_mag,
             c = 'grey', label = "Variable stars")



plt.xlabel("Color"); plt.ylabel("Absolute magnitude"); plt.legend()
plt.title("Stellar properties in NE galaxy")
plt.show()
    
    
    

    
