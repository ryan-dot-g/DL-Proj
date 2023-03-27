# module imports
import numpy as np 
import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os 
import pandas as pd
# from tqdm import tqdm
# from astropy.timeseries import LombScargle
ddir = 'C:\\Users\\rgray\\OneDrive\\ryan\\Uni\\2023 sem 1\\PHYS3080\\Assignments\\DL-Proj\\DATA\\' 

# define default plot settings
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
matplotlib.rcParams['savefig.dpi']= 300             #72 

ddir += '/Variable_Star_Data/'

'''         INDIVIDUAL STAR SECTION         '''

fname = 'BackS000011.csv' # star fname'

data = pd.read_csv(ddir+fname) 
time, flux = data.Time, data.NormalisedFlux 
dt = np.median(np.diff(time))
nqlim = 1/2 * dt # nyquist limit

plt.plot(time,flux,'.',markersize=16)
plt.xlabel('Time (h)')
plt.ylabel('Relative Flux')
plt.title("Variable star period-luminosity")
plt.show()

from astropy.timeseries import LombScargle
LS = LombScargle(time,flux) # initialize a Lomb-Scargle algorithm from Astropy
freqs = np.linspace(1/100,0.45,10000) # frequency grid shouldn't go higher than Nyquist limit
power = LS.power(freqs) # calculate LS power
print('Best period: %.2f h' % (1/freqs[np.argmax(power)]))
plt.plot(freqs,power)
plt.xlabel('Frequency (c/h)')
plt.ylabel('LS Power')
plt.show()

