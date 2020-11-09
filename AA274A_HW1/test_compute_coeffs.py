#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:43:35 2020

@author: paschoeto
"""
import numpy as np

from P1_differential_flatness import State, compute_traj_coeffs, compute_traj, compute_controls, rescale_V, compute_arc_length, compute_tau

initial_state = State(0,0,0.5,-np.pi/2)
final_state = State(5,5,0.5,-np.pi/2)
tf = 15

coeffs = compute_traj_coeffs(initial_state,final_state,tf)

t, traj = compute_traj(coeffs, tf, 10)

V, om = compute_controls(traj)

V_tilde = rescale_V(V,om,0.5,1)

print(V_tilde)
#s = compute_arc_length(V,t)

#print(V_tilde.shape)
#print(s.shape)
#tau = compute_tau(V_tilde, s)

#print(tau)



