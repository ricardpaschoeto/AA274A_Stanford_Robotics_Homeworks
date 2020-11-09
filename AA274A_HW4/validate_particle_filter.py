#!/usr/bin/env python

import numpy as np
import numpy.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from ExtractLines import ExtractLines
import maze_sim_parameters as params
from particle_filter import MonteCarloLocalization
from ekf import EkfLocalization
from validate_ekf import validate_ekf_transition_update
import pickle

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
PF_PICKLE = "pf_validation.pickle"

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

### PROBLEM 3

def validate_transition_model():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation_pf = load_pickle(PF_PICKLE)
    x0 = validation_pf['x0']
    xs_validation = validation_pf['transition_model_validation']
    dt = validation_run['t'][1] - validation_run['t'][0]

    # Initialize MC localization
    np.random.seed(1234)
    mcl = MonteCarloLocalization(validation_pf['x0'],
                                 10. * NoiseParams['R'],
                                 params.MapParams,
                                 validation_run['tf_base_to_camera'],
                                 NoiseParams['g'])

    xs = mcl.transition_model(np.random.multivariate_normal(validation_run['controls'][0],
                                                            10. * dt * NoiseParams['R'],
                                                            (x0.shape[0],)), dt)

    if np.linalg.norm(xs_validation - xs, axis=1).sum() >= 1e-6:
        print("Got MonteCarloLocalization.transition_model() particles:\n")
        print(xs)
        print("\nvs. the expected particles\n")
        print(xs_validation)
        return False

    print("MonteCarloLocalization.transition_model() seems to be correct")
    return True

def validate_predicted_measurements():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation_pf = load_pickle(PF_PICKLE)
    x0 = validation_pf['x0']
    xs_input = validation_pf['x_input']
    dt = validation_run['t'][1] - validation_run['t'][0]
    ground_truth_hs = validation_pf['predicted_measurements_validation']

    # Initialize MC localization
    np.random.seed(1234)
    mcl = MonteCarloLocalization(xs_input,
                                 10. * NoiseParams['R'],
                                 params.MapParams,
                                 validation_run['tf_base_to_camera'],
                                 NoiseParams['g'])

    hs = mcl.compute_predicted_measurements()

    if np.linalg.norm(hs - ground_truth_hs, axis=1).sum() >= 1e-2:
        print("Got MonteCarloLocalization.compute_predicted_measurements() output:\n")
        print(hs)
        print("\nvs. the expected values\n")
        print(ground_truth_hs)
        return False

    print("MonteCarloLocalization.compute_predicted_measurements() seems to be correct")
    return True

    # with open("validation_pf.pickle", "wb") as f:
    #     validation_pf['predicted_measurements_validation'] = hs
    #     pickle.dump(validation_pf, f)

def validate_compute_innovations():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation_pf = load_pickle(PF_PICKLE)
    x0 = validation_pf['x0']
    xs_input = validation_pf['x_input']
    dt = validation_run['t'][1] - validation_run['t'][0]
    scans = validation_run['scans']
    ground_truth_vs = validation_pf['predicted_compute_innovations']

    # Initialize MC localization
    np.random.seed(1234)
    mcl = MonteCarloLocalization(xs_input,
                                 10. * NoiseParams['R'],
                                 params.MapParams,
                                 validation_run['tf_base_to_camera'],
                                 NoiseParams['g'])

    alpha, r, Q_raw, _, _ = ExtractLines(scans[0,-1,:],
                                         scans[1,-1,:],
                                         LineExtractionParams,
                                         NoiseParams['var_theta'],
                                         NoiseParams['var_rho'])
    z_raw = np.vstack((alpha, r))
    vs = mcl.compute_innovations(z_raw, np.array(Q_raw))

    if np.linalg.norm(vs - ground_truth_vs, axis=1).sum() >= 1e-3:
        print("Got MonteCarloLocalization.compute_innovations() output:\n")
        print(vs)
        print("\nvs. the expected values\n")
        print(ground_truth_vs)
        return False

    print("MonteCarloLocalization.compute_innovations() seems to be correct")
    return True

    # with open("validation_pf.pickle", "wb") as f:
    #     validation_pf['predicted_compute_innovations'] = vs
    #     pickle.dump(validation_pf, f)

def validate_resample():
    # Load pickle
    validation_run = load_pickle(EKF_PICKLE)
    validation_pf = load_pickle(PF_PICKLE)
    x0 = validation_pf['x0']
    xs_input = validation_pf['x_input']
    ws_input = validation_pf['w_input']
    xs_validation = validation_pf['resample_validation']['xs']
    ws_validation = validation_pf['resample_validation']['ws']

    # Initialize MC localization
    np.random.seed(1234)
    mcl = MonteCarloLocalization(x0,
                                 10. * NoiseParams['R'],
                                 params.MapParams,
                                 validation_run['tf_base_to_camera'],
                                 NoiseParams['g'])

    mcl.resample(xs_input, ws_input)

    if np.linalg.norm(xs_validation - mcl.xs, axis=1).sum() >= 1e-6:
        print("Got MonteCarloLocalization.resample() particles:\n")
        print(mcl.xs)
        print("\nvs. the expected particles\n")
        print(xs_validation)
        return False
    if np.linalg.norm(ws_validation - mcl.ws) >= 1e-6:
        print("Got MonteCarloLocalization.resample() weights:\n")
        print(mcl.ws)
        print("\nvs. the expected weights\n")
        print(ws_validation)
        return False

    print("MonteCarloLocalization.resample() seems to be correct")
    return True
        
    # validation_pf["resample_validation"] = {"xs": mcl.xs, "ws": mcl.ws}
    # with open(fname, "wb") as f:
    #     pickle.dump(validation_pf, f)

def validate_mc_localization(show_plot=True):
    NUM_PARTICLES=100

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
    x0 = np.tile(validation_run['states'][0], (NUM_PARTICLES, 1))

    # Initialize MC localization
    np.random.seed(1234)
    mcl = MonteCarloLocalization(x0,
                                 10. * NoiseParams['R'],
                                 params.MapParams,
                                 validation_run['tf_base_to_camera'],
                                 NoiseParams['g'])

    X = np.zeros((T, NUM_PARTICLES, 3))
    W = np.zeros((T, NUM_PARTICLES))

    # Iterate over states
    mcl_states = np.zeros((T, mcl.x.shape[0]))
    mcl_states[0] = mcl.x
    scan_states = np.zeros((T_scans, mcl.x.shape[0]))
    scan_idx = 0
    X[0] = mcl.xs
    W[0] = mcl.ws
    for i in range(T - 1):
        t1 = t[i+1]
        t0 = t[i]

        # Iterate over scans
        while scan_idx < T_scans and t_scans[scan_idx] < t1:
            # Transition update
            mcl.transition_update(u[i], t_scans[scan_idx] - t0)
            t0 = t_scans[scan_idx]

            # Measurement update
            alpha, r, Q_raw, _, _ = ExtractLines(scans[0,scan_idx,:],
                                                 scans[1,scan_idx,:],
                                                 LineExtractionParams,
                                                 NoiseParams['var_theta'],
                                                 NoiseParams['var_rho'])
            z_raw = np.vstack((alpha, r))
            mcl.measurement_update(z_raw, Q_raw)
            scan_states[scan_idx] = mcl.x
            scan_idx += 1

        # Transition update
        mcl.transition_update(u[i], t1 - t0)
        mcl_states[i+1] = mcl.x

        X[i+1] = mcl.xs
        W[i+1] = mcl.ws

    # Plot
    plt.plot(mcl_states[:,0], mcl_states[:,1], label="MCL", color="red")
    plt.scatter(scan_states[:,0], scan_states[:,1], marker = "x", label="measurement update", color="blue")
    if show_plot:
        plt.legend(loc=0)
        plt.savefig("mc_localization.png")
        print("Plot saved to mc_localization.png")

    # with open("validation_pf.pickle", "wb") as f:
    #     pickle.dump({"Xs": X, "Ws": W}, f)


if __name__ == '__main__':
    ### PROBLEM 1
    #validate_mc_localization()

    ## Subcomponent validation
    #validate_transition_model()
    validate_predicted_measurements()
    #validate_compute_innovations()
    validate_resample()
