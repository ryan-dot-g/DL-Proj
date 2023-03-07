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

class Galaxy():
    def __init__(self,stars,camera):
        self.camera = camera
        self.stars = stars
        
BOTTOM = [
    [ [-45,20], [20,45] ],
    [ [-45,45], [-45,45] ],
    [3, 14]
    ]

FRONT = [
    [ [-45,45] ],
    [ [-45,45] ],
    [ 10 ]
    ]

CAMERA_CLUSTERS = {"Bottom":BOTTOM, "Front":FRONT}

from sklearn.cluster import KMeans

for camera, cameraData in CAMERA_CLUSTERS.items():
    stars = pd.read_csv(ddir + camera + '\\Star_Data.csv') 
    
    # plot all the stars
    plt.scatter(stars.X,stars.Y, s = 0.01)
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')
    plt.title(f'Star plot ({camera.lower()} camera)')
    plt.show() 
    
    for i in range(len(cameraData[0])):
        xmin,xmax = cameraData[0][i]
        ymin,ymax = cameraData[1][i]
        ngalaxies = cameraData[2][i]
        
        starCluster = stars[ (xmin <= stars.X) * (stars.X < xmax) *
                             (ymin <= stars.Y) * (stars.Y < ymax) ]
        
        # zipped (x,y) coords
        R = np.array([ i for i in zip(*[starCluster.X,starCluster.Y]) ]) 
        km = KMeans(n_clusters = ngalaxies)
        km.fit(R)
    
        cmap = plt.cm.get_cmap('tab20')
        plt.scatter( starCluster.X, starCluster.Y, c = km.labels_, s = 0.01,
                    cmap = cmap, alpha = 0.5)
        plt.title(f"{camera} camera, cluster {i}")
        plt.xlabel("x (pix)"); plt.ylabel("y (pix")
        plt.axis('equal')
        plt.show()