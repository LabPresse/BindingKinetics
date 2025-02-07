
# Import standard modules
if __name__ == "__main__":
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
import os
import sys
import copy
import time
import random
import numpy as np
import numba as nb
import h5py
import matplotlib.pyplot as plt
from scipy import stats, sparse
from types import SimpleNamespace
from dict2hdf5 import dict_to_h5


# Define sampling function
@nb.njit
def sample_states(pi0, pis, num_frames):

    # Prepare variables
    pi0 = np.ascontiguousarray(pi0)
    pis = np.ascontiguousarray(pis)
    states = np.zeros(num_frames, dtype=np.int32)

    # Sample states
    s = np.searchsorted(np.cumsum(pi0), np.random.rand())
    states[0] = s
    for n in range(1, num_frames):
        s = np.searchsorted(np.cumsum(pis[s, :]), np.random.rand())
        states[n] = s

    # Return
    return states

# Calculate partitions function
@nb.njit(cache=True)
def nb_calculate_partitions(n,k):
    if(k==1):
        return [[n]]
    if(n==0):
        return [[0]*k]
    return [ g2 for x in range(n+1) for g2 in [ u+[n-x] for u in nb_calculate_partitions(x,k-1) ] ]
def calculate_partitions(n, k):
    """
    How many ways can you put n identical objects into k identical bins?
    """
    partitions = nb_calculate_partitions(n, k)
    partitions = np.array(partitions, dtype=int)
    return partitions


# Define micro2macro numba
@nb.njit(cache=True)
def nb_micro2macro(k_on_eff, k_off_eff, k_photo_eff, partitions):

    # Get constants
    num_max = np.sum(partitions[0, :])
    num_macro, num_micro = partitions.shape

    # Initialize output
    vals = []
    rows = []
    cols = []

    # Loop through partitions
    for i in range(partitions.shape[0]-1, -1, -1):

        # Initial state probability
        P = np.prod(.1 ** partitions[i, :-1])
        vals.append(P)
        rows.append(num_macro)
        cols.append(i)

        # Get variables
        pop1 = partitions[i, :]
        can_bind = np.zeros(num_micro)
        can_bind[pop1 != num_max] = 1
        esc_rate = (
            -np.diag(k_photo_eff) @ pop1  # Photo-transitions
            + (k_on_eff @ can_bind)       # Binding
            + (k_off_eff @ pop1)          # Unbinding
        )
        if esc_rate == 0:
            vals.append(1)
            rows.append(i)
            cols.append(i)
            continue

        # Set self-transition
        selftrans = np.exp(-esc_rate)
        vals.append(selftrans)
        rows.append(i)
        cols.append(i)

        # Loop over allowed transitions
        diffs = partitions - pop1
        allowed = np.where(np.sum(np.abs(diffs), axis=1) <= 2)[0]
        for j in allowed:
            if j == i:
                continue

            # Initialize probability
            P = (1 - selftrans) / esc_rate

            # Find which transition occured
            diff = diffs[j, :]
            s_old = np.where(diff == -1)[0][0]
            s_new = np.where(diff == 1)[0][0]
            if s_old == num_micro - 1:
                # Binding
                P *= k_on_eff[s_new]
            elif s_new == num_micro - 1:
                # Unbinding or photobleaching
                P *= pop1[s_old] * (k_off_eff[s_old] + k_photo_eff[s_old, s_new])
            else:
                # Phototransition
                P *= pop1[s_old] * k_photo_eff[s_old, s_new]
            
            # Add to output
            if P > 0:
                vals.append(P)
                rows.append(i)
                cols.append(j)
    
    # Return output
    return vals, rows, cols

# Define micro2macro
def micro2macro(k_on, k_off, k_photo, dt, c, l, partitions):


    # Set up constants
    k_on_eff = dt * k_on * c
    k_off_eff = dt * k_off
    k_photo_eff = dt * k_photo
    k_photo_eff[:, -1] *= l
    for k in range(k_photo_eff.shape[0]):
        k_photo_eff[k, k] = - (np.sum(k_photo_eff[k, :]) - k_photo_eff[k, k])
    num_macro = partitions.shape[0]

    # Run numba function
    vals, rows, cols = nb_micro2macro(k_on_eff, k_off_eff, k_photo_eff, 1.0*partitions)
    
    # Create sparse matrix
    pi = sparse.coo_matrix((vals, (rows, cols)), shape=(num_macro+1, num_macro)).tocsc()

    # Return output
    return pi



#### SIMULATE DATA ####

# Simulate data function
def simulate_data(parameters=None, seed=None, **kwargs):

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Set parameters
    print("Setting parameters")
    default_parameters = {
        # Variables
        "s": None,
        "pi0": None,
        "k_on": None,
        "k_off": None,
        "k_pb": None,
        "k_photo": None,
        "mu_flor": 400,
        "mu_back": 0,
        "sigma": 2500,
        "gain": 10,
        # Experiment
        "dt": 1,
        "settings": None,
        "laserpower": None,
        "concentration": None,
        "partitions": None,
        # Numbers
        "num_rois": 1000,
        "num_frames": 8000,
        "num_micro": 3,
        "num_macro": None,
        "num_max": 10,
    }
    if parameters is None:
        parameters = {}
    parameters = {**default_parameters, **parameters, **kwargs}

    # Set up variables
    print("Setting up variables")
    variables = SimpleNamespace(**parameters)
    s = variables.s
    pi0 = variables.pi0
    k_on = variables.k_on
    k_off = variables.k_off
    k_pb = variables.k_pb
    k_photo = variables.k_photo
    mu_flor = variables.mu_flor
    mu_back = variables.mu_back
    sigma = variables.sigma
    gain = variables.gain
    dt = variables.dt
    settings = variables.settings
    laserpower = variables.laserpower
    concentration = variables.concentration
    partitions = variables.partitions
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_micro = variables.num_micro
    num_macro = variables.num_macro
    num_max = variables.num_max
    
    # Set experiment parameters
    print("Setting experiment parameters")
    if concentration is None:
        concentration = np.repeat([1.0, 2.0, 5.0, 10.0], int(num_rois/4+1))[:num_rois]
        #concentration = 5*np.ones(num_rois, dtype=int)
    if laserpower is None:
        laserpower = np.tile([50, 100], int(num_rois/2+1))[:num_rois]
    if settings is None:
        settings = np.unique(np.vstack([concentration, laserpower]).T, axis=0)
    variables.concentration = concentration
    variables.laserpower = laserpower
    variables.settings = settings

    # Set state space
    print("Setting state space")
    partitions = calculate_partitions(num_max, num_micro)
    num_macro = partitions.shape[0]
    variables.num_macro = num_macro
    variables.partitions = partitions

    # Set initial state
    print("Setting initial state")
    if pi0 is None:
        pi0 = np.zeros(num_macro)
        pi0[0] = 1
    variables.pi0 = pi0

    # Set transition rates
    print("Setting transition rates")
    if k_on is None:
        k_on = 10/(dt*num_frames)
    if np.isscalar(k_on):
        k_on = k_on * np.ones(num_micro, dtype=float)
        k_on[1:] = 0
    if k_off is None:
        k_off = 20/(dt*num_frames)
    if np.isscalar(k_off):
        k_off = k_off * np.ones(num_micro, dtype=float)
        k_off[1:] = 0
    if k_pb is None:
        k_pb = 10/(dt*num_frames)
    if k_photo is None:
        k_photo = np.zeros((num_micro, num_micro), dtype=float)
        if num_micro == 2:
            k_photo[0, 1] = k_pb
        elif num_micro == 3:
            k_photo[0, 2] = k_pb                 # Bleach
            k_photo[0, 1] = 1/(dt*num_frames)    # Blink
            k_photo[1, 0] = 100/(dt*num_frames)  # Unblink
        for k in range(num_micro):
            k_photo[k, k] = - np.sum(k_photo[k, :])
    variables.k_on = k_on
    variables.k_off = k_off
    variables.k_pb = k_pb
    variables.k_photo = k_photo

    # Create data and states
    print("Creating data and states")
    data = np.zeros((num_rois, num_frames), dtype=float)
    states = np.zeros((num_rois, num_frames), dtype=int)
    for i, (c, l) in enumerate(settings):
        print(f"--{i}/{len(settings)}")
        # Get indices of settings
        idx = np.where((concentration == c) & (laserpower == l))[0]
        # Calculate transition probabilities
        pi = micro2macro(k_on, k_off, k_photo, dt, c, l, partitions=partitions)
        # Loop over ROIs
        for r in idx:
            # Sample trajectories until at least one is not all zeros
            trys = 0
            while trys < 3:
                states_r = sample_states(pi0, pi.todense(), num_frames)
                if np.sum(states_r > 0) > 0:
                    states[r, :] = states_r
                    break
            for k in range(partitions.shape[0]):
                idx = states_r == k
                mu = mu_flor*l*partitions[k, 0] + mu_back
                data[r, idx] = stats.norm.rvs(loc=mu, scale=sigma, size=np.sum(idx))
    
    # Set variables
    variables.s = states

    # Return
    return data, variables

# Main script
if __name__ == "__main__":
    """
    Base is {
        k_on: 0.000001,
        k_off: 0.000025,
        k_photo: np.array([
            [-0.0000001, 0.0000001],
            [0, 0],
        ]),
        num_micro: 2,
    }
    We do 3 tests:
    1. Vary k_on in [1, 2, 5] * 1e-6
    2. Vary k_off in [1, 2, 5] * 1e-5
    3. Vary k_photo in [1, 2, 5] * 1e-7
    """

    # Set base parameters
    base_parameters = {
        'dt': 124,
        'num_rois': 2000,
        'k_on': 0.000001,
        'k_off': 0.000025,
        'k_photo': np.array([
            [-0.0000001, 0.0000001],
            [0, 0],
        ]),
        'num_micro': 2,
    }

    # Set up list of modifications
    mods = [
        {'k_on': 0.000001},
        {'k_on': 0.000002},
        {'k_on': 0.000005},
        {'k_off': 0.00001},
        {'k_off': 0.00002},
        {'k_off': 0.00005},
        {'k_photo': np.array([
            [-0.0000001, 0.0000001],
            [0, 0],
        ])},
        {'k_photo': np.array([
            [-0.0000002, 0.0000002],
            [0, 0],
        ])},
        {'k_photo': np.array([
            [-0.0000005, 0.0000005],
            [0, 0],
        ])},
    ]

    # Loop over parameters
    for ID in range(len(mods)):

        # Get modification
        mod = mods[ID]
        gt_parameters = {**base_parameters, **mod}
        savename = "simulated_" + "_".join([
            f"kon={int(gt_parameters['k_on']*1e6)}e-6",
            f"koff={int(gt_parameters['k_off']*1e5)}e-5",
            f"kphoto={int(gt_parameters['k_photo'][0, 1]*1e7)}e-7",
            f"Nstates={gt_parameters['num_micro']}",
        ])

        print(savename)
        continue

        # Simulate data
        data, variables = simulate_data(parameters=gt_parameters, seed=42)

        # Plot data
        fig, ax = plt.subplots(3, 1, sharex=True, squeeze=False)
        plt.ion()
        plt.show()
        ax[0, 0].plot(data[0, :])
        ax[1, 0].plot(data[1, :])
        ax[2, 0].plot(data[2, :])
        plt.pause(1)

        # Extract parameters from variables
        parameters = {
            "dt": variables.dt,
            "settings": variables.settings,
            "laserpower": variables.laserpower,
            "concentration": variables.concentration,
        }

        # Save data
        path = f"{os.environ['DATAPATH']}/Binding/"
        output = {"data": data, "groundtruth": variables.__dict__, **parameters}
        with h5py.File(path + savename + ".h5", "w") as h5:
            dict_to_h5(h5, output)

    print("Done!")

