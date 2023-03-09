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


'''                 BEGIN DISTANT GALAXY CALIBRATION SECTION            '''
camera = "Bottom";

flashes = pd.read_csv(ddir + '\\Flash_Data.csv')




fuzzies = pd.read_csv(ddir + camera + '\\Distant_Galaxy_Data.csv') 