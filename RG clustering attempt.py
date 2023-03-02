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

CAMERAS = ['Top','Bottom','Front','Back','Left','Right']
nclusters = [8, 15, 15, 15, 15, 15]
nclustDict = dict( zip(*[CAMERAS,nclusters]) )

# set data directory and read data
camera = 'Top'
stars = pd.read_csv(ddir + camera + '\\Star_Data.csv') 
print(stars.keys()) # this tells us what column names we have

# plot all the stars
plt.scatter(stars.X,stars.Y, s = 0.01)
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.title(f'Star plot ({camera.lower()} camera)')
plt.show() 

'''                 BEGIN CLUSTERING SECTION            '''
from sklearn.cluster import KMeans,OPTICS

stars = stars[ (stars.X > -5) * (stars.Y < -25) ]

R = np.array([ i for i in zip(*[stars.X,stars.Y]) ]) # zipped (x,y) coords
km = KMeans(n_clusters = nclustDict[camera])
km.fit(R)


cmap = plt.cm.get_cmap('tab20')
plt.scatter( stars.X, stars.Y, c = km.labels_, s = 0.01,
            cmap = cmap, alpha = 0.5)
plt.axis('equal')
plt.show()