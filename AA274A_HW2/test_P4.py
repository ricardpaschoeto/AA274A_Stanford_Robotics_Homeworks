# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:58:59 2020

@author: paschoeto
"""

import numpy as np
import matplotlib.pyplot as plt
from P2_rrt import *
from P4_bidirectional_rrt import *

plt.rcParams['figure.figsize'] = [7, 7] # Change default figure size

MAZE = np.array([
    (( 5, 5), (-5, 5)),
    ((-5, 5), (-5,-5)),
    ((-5,-5), ( 5,-5)),
    (( 5,-5), ( 5, 5)),
    ((-5, 2), (-1, 2)),
    ((-1, 2), (-1,-1)),
    (( 0, 2), ( 0,-1)),
    (( 0, 2), ( 5, 2))
])

#grrt = GeometricRRTConnect([-5,-5], [5,5], [-4,-4], [4,4], MAZE)
#grrt.solve(1.0, 2000)

drrt = DubinsRRTConnect([-5,-5,0], [5,5,2*np.pi], [-4,-4,0], [4,4,np.pi/2], MAZE, .5)
drrt.solve(1.0, 1000)