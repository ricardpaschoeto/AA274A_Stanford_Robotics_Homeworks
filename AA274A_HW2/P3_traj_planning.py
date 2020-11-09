import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import compute_controls, compute_arc_length, rescale_V, rescale_om, compute_tau, interpolate_traj, State
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import TrajectoryTracker

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########

        if (t < (self.traj_controller.traj_times[-1] - self.t_before_switch)):
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            return self.pose_controller.compute_control(x, y, th, t)

        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    path = np.asarray(path)
    x = path[:,0]
    y = path[:,1]
    
    t_path = np.zeros((len(path[:,0])))
    for ii in range(0,len(path[:,0])-1):
        delta_t = np.linalg.norm(path[ii,:]-path[ii+1,:])/V_des
        t_path[ii+1] = t_path[ii] + delta_t
        
    t_smoothed = np.arange(0,t_path[-1], dt)
    
    spl_x = scipy.interpolate.splrep(t_path, x, k=3, s=alpha)
    spl_y = scipy.interpolate.splrep(t_path, y, k=3, s=alpha)

    
    splv_x = scipy.interpolate.splev(t_smoothed, spl_x, der=0)
    splv_y = scipy.interpolate.splev(t_smoothed, spl_y, der=0)
    
    splv_xd = scipy.interpolate.splev(t_smoothed, spl_x, der=1)
    splv_yd = scipy.interpolate.splev(t_smoothed, spl_y, der=1)
    
    theta = np.arctan2(splv_yd,splv_xd)
    
    splv_xdd = scipy.interpolate.splev(t_smoothed, spl_x, der=2)
    splv_ydd = scipy.interpolate.splev(t_smoothed, spl_y, der=2)

    
    traj_smoothed = np.array([splv_x, splv_y, theta, splv_xd,splv_yd, splv_xdd,splv_ydd]).T

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    s_f = State(x=traj[-1,0], y=traj[-1,1],V=V_max, th=traj[-1,2])
    V, om = compute_controls(traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde,dt, s_f)
    
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
