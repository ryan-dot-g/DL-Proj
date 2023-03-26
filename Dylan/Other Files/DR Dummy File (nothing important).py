#from __future__ import print_function, division

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:55:02 2023

@author: dylan
"""

### RANDOM DUMMY TEST FILE ###
""" RANDOM DUMMY TEST FILE """


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


import numpy as np
from scipy.optimize import fmin
import math

def f(x):
    exp = (math.pow(x[0], 2) + math.pow(x[1], 2)) * -1
    return math.exp(exp) * math.cos(x[0] * x[1]) * math.sin(x[0] * x[1])

fmin(f,np.array([0,0]))

