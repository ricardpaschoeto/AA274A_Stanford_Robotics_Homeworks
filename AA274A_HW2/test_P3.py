# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:08:27 2020

@author: paschoeto
"""

from P1_astar import DetOccupancyGrid2D, AStar

import numpy as np
from P3_traj_planning import compute_smoothed_traj
from utils import generate_planning_problem
import matplotlib.pyplot as plt

width = 100
height = 100
num_obs = 25
min_size = 5
max_size = 30

occupancy, x_init, x_goal = generate_planning_problem(width, height, num_obs, min_size, max_size)

astar = AStar((0, 0), (width, height), x_init, x_goal, occupancy)
if not astar.solve():
    print("No path found")
    
## ===============================================

V_des = 0.3  # Nominal velocity
alpha = 4.8# Smoothness parameter
dt = 0.05

traj_smoothed, t_smoothed = compute_smoothed_traj(astar.path, V_des, alpha, dt)

fig = plt.figure()
astar.plot_path(fig.number)
def plot_traj_smoothed(traj_smoothed):
    plt.plot(traj_smoothed[:,0], traj_smoothed[:,1], color="red", linewidth=2, label="solution path", zorder=10)
plot_traj_smoothed(traj_smoothed)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
plt.show()