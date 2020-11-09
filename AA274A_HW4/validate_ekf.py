#!/usr/bin/env python

import numpy as np
import numpy.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from ExtractLines import ExtractLines
import maze_sim_parameters as params
from ekf import EkfLocalization, EkfSlam

### PARAMETERS ASSOCIATED WITH `validation_run.pickle`

LineExtractionParams = {'MIN_SEG_LENGTH': 0.1,             # minimum length of each line segment (m)
                        'LINE_POINT_DIST_THRESHOLD': 0.05, # max distance of pt from line to split
                        'MAX_P2P_DIST': 0.4,               # max distance between two adjent pts within a segment
                        'MIN_POINTS_PER_SEGMENT': 3}       # minimum number of points per line segment

NoiseParams = {'Sigma0': 0.01*np.eye(3),  # initial state covariance (x0 comes from ground truth; nonzero in case of timing mismatch)
               'R': 0.1*np.eye(2),    # control noise covariance (corresponding to dt = 1 second)
               'var_theta': 0.1,      # laser scan noise variance in theta measurement (per point)
               'var_rho': 0.1,        # laser scan noise variance in rho measurement (per point)
               'g': 3.,               # validation game (essentially maximum z-score)
               'std_alpha': 0.1,     # noisy map stdev in alpha for EKF_SLAM (per line)
               'std_r': 0.2}         # noisy map stdev in r for EKF_SLAM (per line)

EKF_PICKLE = "ekf_validation.pickle"

def load_pickle(fname):
    import pickle
    with open(fname, "rb") as f:
        return pickle.load(f)

### PROBLEM 1

def validate_ekf_transition_update(show_plot=True):
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    ground_truth_states = validation_run['states']
    u = validation_run['controls']
    dt = validation_run['t'][1:] - validation_run['t'][:-1]
    T = validation_run['t'].shape[0]

    # Initialize EKF localization
    ekf_loc = EkfLocalization(ground_truth_states[0],
                              NoiseParams['Sigma0'],
                              NoiseParams['R'],
                              params.MapParams,
                              validation_run['tf_base_to_camera'],
                              NoiseParams['g'])

    # Simulate controls
    open_loop_states = np.zeros((T, ekf_loc.x.shape[0]))
    open_loop_states[0] = ekf_loc.x
    for i in range(T - 1):
        ekf_loc.transition_update(u[i], dt[i])
        open_loop_states[i+1] = ekf_loc.x

    # Plot
    plt.clf()
    plt.plot(ground_truth_states[:,0], ground_truth_states[:,1], label="ground truth", color="black")
    plt.plot(open_loop_states[:,0], open_loop_states[:,1], label="open loop", color="green")
    plt.axis("equal")
    if show_plot:
        plt.legend(loc=0)
        plt.savefig("ekf_open_loop.png")
        print("Plot saved to ekf_open_loop.png")

def validate_ekf_localization(show_plot=True):
    # Plot open loop
    validate_ekf_transition_update(False)

    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    u = validation_run['controls']
    t = validation_run['t']
    T = t.shape[0]
    t_scans = validation_run['t_scans']
    T_scans = t_scans.shape[0]
    scans = validation_run['scans']

    # Initialize EKF localization
    ekf_loc = EkfLocalization(validation_run['states'][0],
                              NoiseParams['Sigma0'],
                              NoiseParams['R'],
                              params.MapParams,
                              validation_run['tf_base_to_camera'],
                              NoiseParams['g'])

    # Iterate over states
    ekf_states = np.zeros((T, ekf_loc.x.shape[0]))
    ekf_states[0] = ekf_loc.x
    scan_states = np.zeros((T_scans, ekf_loc.x.shape[0]))
    scan_idx = 0
    for i in range(T - 1):
        t1 = t[i+1]
        t0 = t[i]

        # Iterate over scans
        while scan_idx < T_scans and t_scans[scan_idx] < t1:
            # Transition update
            ekf_loc.transition_update(u[i], t_scans[scan_idx] - t0)
            t0 = t_scans[scan_idx]

            # Measurement update
            alpha, r, Q_raw, _, _ = ExtractLines(scans[0,scan_idx,:],
                                                 scans[1,scan_idx,:],
                                                 LineExtractionParams,
                                                 NoiseParams['var_theta'],
                                                 NoiseParams['var_rho'])
            z_raw = np.vstack((alpha, r))
            ekf_loc.measurement_update(z_raw, Q_raw)
            scan_states[scan_idx] = ekf_loc.x
            scan_idx += 1

        # Transition update
        ekf_loc.transition_update(u[i], t1 - t0)
        ekf_states[i+1] = ekf_loc.x

    # Plot
    plt.plot(ekf_states[:,0], ekf_states[:,1], label="EKF (known map)", color="red")
    plt.scatter(scan_states[:,0], scan_states[:,1], marker = "x", label="measurement update", color="blue")
    if show_plot:
        plt.legend(loc=0)
        plt.savefig("ekf_localization.png")
        print("Plot saved to ekf_localization.png")

def validate_localization_transition_model():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    u = validation_run['controls']
    T = validation_run['t'].shape[0]
    validation = validation_run['transition_model_validation']

    # Initialize EKF localization
    ekf_loc = EkfLocalization(validation_run['states'][0],
                              NoiseParams['Sigma0'],
                              NoiseParams['R'],
                              params.MapParams,
                              validation_run['tf_base_to_camera'],
                              NoiseParams['g'])

    # Simulate controls
    for i in range(T):
        g, Gx, Gu = ekf_loc.transition_model(u[i], 0.1)
        g_ref, Gx_ref, Gu_ref = validation[i]
        if np.linalg.norm(g - g_ref) + np.linalg.norm(Gx - Gx_ref) + np.linalg.norm(Gu - Gu_ref) > 1e-2:
            print("At state x = {0} with u = {1} and dt = {2} got EkfLocalization.transition_model output:\n".format(ekf_loc.x, u, 0.1))
            print(g)
            print(Gx)
            print(Gu)
            print("\nvs. the expected values\n")
            print(g_ref)
            print(Gx_ref)
            print(Gu_ref)
            return False
    
    print("EkfLocalization.transition_model() seems to be correct")
    return True

def validate_localization_compute_predicted_measurements():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation = validation_run['compute_predicted_measurements_validation']

    # Initialize EKF localization
    ekf_loc = EkfLocalization(validation_run['states'][0],
                              NoiseParams['Sigma0'],
                              NoiseParams['R'],
                              params.MapParams,
                              validation_run['tf_base_to_camera'],
                              NoiseParams['g'])

    # Compare measurements
    hs, Hs = ekf_loc.compute_predicted_measurements()
    for j in range(ekf_loc.map_lines.shape[1]):
        h, Hx = hs[:,j], Hs[j]
        h_ref, Hx_ref = validation[j]
        if np.linalg.norm(h - h_ref) + np.linalg.norm(Hx - Hx_ref) > 1e-3:
            print("At state x = {0} with m = {1} got EkfLocalization.compute_predicted_measurements:\n".format(ekf_loc.x, ekf_loc.map_lines[:,j]))
            print(h)
            print(Hx)
            print("\nvs. the expected values\n")
            print(h_ref)
            print(Hx_ref)
            return False

    print("EkfLocalization.compute_predicted_measurements() seems to be correct")
    return True

def validate_localization_compute_innovations():
    # Test transition_model() and predicted_measurements() first
    if not validate_localization_transition_model() or \
       not validate_localization_compute_predicted_measurements():
        print("Validation of compute_innovations cannot proceed.")
        return False

    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation_input = validation_run['compute_innovations_validation_input']
    validation = validation_run['compute_innovations_validation']
    scans = validation_run['scans']
    T_scans = validation_run['t_scans'].shape[0]

    # Initialize EKF localization
    ekf_loc = EkfLocalization(validation_run['states'][0],
                              NoiseParams['Sigma0'],
                              NoiseParams['R'],
                              params.MapParams,
                              validation_run['tf_base_to_camera'],
                              NoiseParams['g'])

    for i in range(T_scans):
        ekf_loc.x, ekf_loc.Sigma = validation_input[i]
        alpha, r, Q_raw, _, _ = ExtractLines(scans[0,i,:],
                                             scans[1,i,:],
                                             LineExtractionParams,
                                             NoiseParams['var_theta'],
                                             NoiseParams['var_rho'])
        z_raw = np.vstack((alpha, r))
        v_list, R_list, H_list = ekf_loc.compute_innovations(z_raw, Q_raw)
        v_list_ref, R_list_ref, H_list_ref = validation[i]
        if len(v_list) != len(v_list_ref) or len(R_list) != len(R_list_ref) or len(H_list) != len(H_list_ref):
            print("You may have an error in EkfLocalization.compute_innovations.")
            print("v", v_list, v_list_ref)
            print("R", R_list, R_list_ref)
            print("H", H_list, H_list_ref)
            return False
        permutation = [np.argmin([np.linalg.norm(R - R_ref) for R_ref in R_list_ref]) for R in R_list]
        v_error = sum([np.linalg.norm(v_list[j] - v_list_ref[k]) for j, k in enumerate(permutation)])
        R_error = sum([np.linalg.norm(R_list[j] - R_list_ref[k]) for j, k in enumerate(permutation)])
        H_error = sum([np.linalg.norm(H_list[j] - H_list_ref[k]) for j, k in enumerate(permutation)])
        if v_error + R_error + H_error > 1e-3:
            print("You may have an error in EkfLocalization.compute_innovations.")
            return False

    print("EkfLocalization.compute_innovations() seems to be correct")
    return True


### PROBLEM 2

def validate_ekf_slam():
    # Plot open loop
    validate_ekf_transition_update(False)

    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    u = validation_run['controls']
    t = validation_run['t']
    T = t.shape[0]
    t_scans = validation_run['t_scans']
    T_scans = t_scans.shape[0]
    scans = validation_run['scans']

    # Prepare initial state
    np.random.seed(1234)
    x0_pose = validation_run['states'][0]
    Sigma0_pose = NoiseParams['Sigma0']

    N_map_lines = params.MapParams.shape[1]
    x0_map = params.MapParams.T.flatten()
    x0_map[4:] = x0_map[4:] + np.vstack((NoiseParams['std_alpha']*np.random.randn(N_map_lines-2),    # first two lines fixed
                                         NoiseParams['std_r']*np.random.randn(N_map_lines-2))).T.flatten()
    Sigma0_map = np.diag(np.concatenate((np.zeros(4),
                                         np.array([[NoiseParams['std_alpha']**2 for i in range(N_map_lines-2)],
                                                   [NoiseParams['std_r']**2 for i in range(N_map_lines-2)]]).T.flatten())))

    # Initialize EKF SLAM
    ekf_slam = EkfSlam(np.concatenate((x0_pose, x0_map)),
                       scipy.linalg.block_diag(Sigma0_pose, Sigma0_map),
                       NoiseParams['R'],
                       validation_run['tf_base_to_camera'],
                       NoiseParams['g'])

    # Iterate over states
    ekf_states = np.zeros((T, ekf_slam.x.shape[0]))
    ekf_states[0] = ekf_slam.x
    scan_states = np.zeros((T_scans, ekf_slam.x.shape[0]))
    scan_idx = 0
    for i in range(T - 1):
        t1 = t[i+1]
        t0 = t[i]

        # Iterate over scans
        while scan_idx < T_scans and t_scans[scan_idx] < t1:
            # Transition update
            ekf_slam.transition_update(u[i], t_scans[scan_idx] - t0)
            t0 = t_scans[scan_idx]
            alpha, r, Q_raw, _, _ = ExtractLines(scans[0,scan_idx,:],
                                                 scans[1,scan_idx,:],
                                                 LineExtractionParams,
                                                 NoiseParams['var_theta'],
                                                 NoiseParams['var_rho'])
            z_raw = np.vstack((alpha, r))
            ekf_slam.measurement_update(z_raw, Q_raw)
            scan_states[scan_idx] = ekf_slam.x
            scan_idx += 1

        # Transition update
        ekf_slam.transition_update(u[i], t1 - t0)
        ekf_states[i+1] = ekf_slam.x

    # Plot
    plt.plot(ekf_states[:,0], ekf_states[:,1], label="EKF (noisy map)", color="orange")
    plt.scatter(scan_states[:,0], scan_states[:,1], marker = "x", label="measurement update", color="blue")
    plt.legend(loc=0)
    plt.savefig("ekf_slam.png")
    print("Plot saved to ekf_slam.png")


if __name__ == '__main__':
    ### PROBLEM 1
    validate_ekf_transition_update()
    validate_ekf_localization()

    ## Subcomponent validation
    validate_localization_transition_model()
    validate_localization_compute_predicted_measurements()
    validate_localization_compute_innovations()

    ### PROBLEM 2

    validate_ekf_slam()
