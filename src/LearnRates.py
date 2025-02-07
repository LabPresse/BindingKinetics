
# Import standard modules TEST
import os
import sys
import h5py
import copy
import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import stats, sparse
from types import SimpleNamespace
from joblib import Parallel, delayed
from src.PlotRates import plot_rates
from src.dict2hdf5 import dict_to_h5, h5_to_dict


### Set up numba functions ###

# Partitions
@nb.njit(cache=True)
def gen(n,k):
    if(k==1):
        return [[n]]
    if(n==0):
        return [[0]*k]
    return [ g2 for x in range(n+1) for g2 in [ u+[n-x] for u in gen(x,k-1) ] ]

# Macrostate transition matrix
@nb.njit(cache=True)
def micro2macro_nb(k_on_eff, k_off_eff, k_photo_eff, partitions):

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

# Likelihood
@nb.njit(cache=True)
def likelihood_nb(data, pi0, pi_vals, pi_rows, pi_colptr, lhood_dict):

    # Initialize variables
    idx = np.zeros(len(pi0), dtype=np.bool_)
    lhood = 0
    p = pi0.copy()
    num_macro = len(p)
    num_frames = len(data)

    # Loop through time levels
    for n in range(num_frames):

        # Set probability of wrong states to 0
        s = data[n]
        idx[lhood_dict[s]] = True
        p[~idx] = 0
        idx[lhood_dict[s]] = False

        # Check if probability is 0
        if np.sum(p) == 0:
            return -np.inf

        # Update likelihood
        lhood += np.log(np.sum(p))

        # Update probability
        p_new = p / np.sum(p)
        p_new[p>0] += 1e-200  # Add small number to positives to avoid vanishing probabilities
        p = p_new

        # Calculate p @ pi
        p_new = np.zeros_like(p)
        for i in range(num_macro):
            j = pi_colptr[i]
            while j < pi_colptr[i+1]:
                k = pi_rows[j]
                p_new[i] += p[k] * pi_vals[j]
                j += 1
        p = p_new
    
    return lhood


### Set up class ###

# Declare class
class RateInference:
    def __init__(self):
        pass
    
    # Define parameters
    PARAMETERS = {

        # Variables
        "P": None,
        "pi0": None,
        "k_on": None,
        "k_off": None,
        "k_photo": None,

        # Priors
        "k_on_shape": 2,
        "k_on_scale": None,
        "k_off_shape": 2,
        "k_off_scale": None,
        "k_photo_shape": 2,
        "k_photo_scale": None,

        # Experiment
        "dt": 1,
        "settings": None,
        "laserpower": None,
        "concentration": None,

        # State space
        "partitions": None,
        "lhood_dict": None,
        "bright_ids": None,
        "microstate_names": None,

        # Numbers
        "num_rois": None,
        "num_frames": None,
        "num_settings": None,
        "num_micro": 3,  # 3 States are: Bright, Dark, Unbound/Bleached
        "num_macro": None,
        "num_max": None,

        # Sampling
        "proposal_shape": 100,
        "parallelize": True,
    }

    # Define calculate_partitions
    @staticmethod
    def calculate_partitions(n, k):
        """
        How many ways can you put n identical objects into k identical bins?
        """
        partitions = gen(n, k)
        partitions = np.array(partitions, dtype=int)
        return partitions

    # Define micro2macro
    @staticmethod
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
        vals, rows, cols = micro2macro_nb(k_on_eff, k_off_eff, k_photo_eff, 1.0*partitions)
        
        # Create sparse matrix
        pi = sparse.coo_matrix((vals, (rows, cols)), shape=(num_macro+1, num_macro)).tocsc()

        # Return output
        return pi

    # Define Forward Sum
    @staticmethod
    def likelihood(data, pi, lhood_dict):
        
        # Extract elements of pi
        pi0 = pi[-1, :].toarray().flatten()
        pis = pi[:-1, :].tocsc()
        pi_vals = pis.data
        pi_rows = pis.indices
        pi_colptr = pis.indptr
        
        # Calculate likelihood
        lhood = likelihood_nb(data, pi0, pi_vals, pi_rows, pi_colptr, lhood_dict)
                
        # Return lhood
        return lhood
    
    # Define Forward Filter Backward Sample
    @staticmethod
    def viterbi(data, pi, lhood_dict):

        # Get constants
        pi0 = pi[-1, :].toarray().flatten()
        pis = pi[:-1, :]
        num_frames = len(data)
        num_macro = pi.shape[1]

        # Create lhood
        lhood_rows = []
        lhood_cols = []
        lhood_vals = []
        for n, s in enumerate(data):
            for i in lhood_dict[s]:
                lhood_rows.append(i)
                lhood_cols.append(n)
                lhood_vals.append(1)
        lhood = sparse.coo_matrix((lhood_vals, (lhood_rows, lhood_cols)), shape=(num_macro, num_frames)).tocsc()

        # Initialize states
        states = np.zeros(num_frames, dtype=np.int32)

        # Forward filter
        forward = sparse.csc_matrix((num_macro, num_frames))
        forward_n = lhood[:, 0].toarray()[:, 0] * pi0
        forward_n /= np.sum(forward_n)
        ids = np.where(forward_n > 0)[0]
        forward[ids, 0] = forward_n[ids]
        for n in range(1, num_frames):
            forward_n = lhood[:, n].toarray()[:, 0] * (pis.T @ forward[:, n - 1].toarray()[:, 0])
            forward_n /= np.sum(forward_n)
            ids = np.where(forward_n > 0)[0]
            forward[ids, n] = forward_n[ids]

        # Backward sample
        s = np.argmax(forward[:, -1])
        states[-1] = s
        for m in range(1, num_frames):
            n = num_frames - m - 1
            backward = forward[:, n].toarray()[:, 0] * pis[:, s].toarray()[:, 0]
            backward /= np.sum(backward)
            s = np.argmax(backward)
            states[n] = s

        return states

    # Define initialize_variables
    @staticmethod
    def initialize_variables(data, parameters=None, **kwargs):
        
        # Set up parameters
        if parameters is None:
            parameters = {}
        parameters = copy.deepcopy(parameters)
        parameters = {**RateInference.PARAMETERS, **parameters, **kwargs}.copy()

        # Set up variables
        variables = SimpleNamespace(**parameters)
        P = variables.P
        pi0 = variables.pi0
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        dt = variables.dt
        settings = variables.settings
        laserpower = variables.laserpower
        concentration = variables.concentration
        partitions = variables.partitions
        lhood_dict = variables.lhood_dict
        num_rois = variables.num_rois
        num_frames = variables.num_frames
        num_settings = variables.num_settings
        num_micro = variables.num_micro
        num_macro = variables.num_macro
        num_max = variables.num_max
        proposal_shape = variables.proposal_shape
        microstate_names = variables.microstate_names
        bright_ids = variables.bright_ids

        # Set up shape
        num_rois, num_frames = data.shape
        variables.num_rois = num_rois
        variables.num_frames = num_frames

        # Set up experimental parameters
        if dt is None:
            dt = 1
        if concentration is None:
            concentration = np.ones(num_rois)
        elif np.isscalar(concentration):
            concentration = np.ones(num_rois) * concentration
        if laserpower is None:
            laserpower = np.ones(num_rois)
        elif np.isscalar(laserpower):
            laserpower = np.ones(num_rois) * laserpower
        settings = np.vstack((concentration, laserpower)).T
        settings = np.unique(settings, axis=0)
        num_settings = len(settings)
        variables.dt = dt
        variables.concentration = concentration
        variables.laserpower = laserpower
        variables.settings = settings
        variables.num_settings = num_settings
        
        # Set up k_on
        if k_on is None:
            if num_micro == 2:
                k_on = 1000/(dt*num_frames) * np.array([1, 0])
            elif num_micro == 3:
                k_on = 1000/(dt*num_frames) * np.array([1, 0, 0])
            elif num_micro == 4:
                k_on = 1000/(dt*num_frames) * np.array([1, 1, 0, 0])
        k_on = k_on * np.ones(num_micro)
        if k_on_scale is None:
            k_on_scale = k_on / k_on_shape
        k_on_shape = k_on_shape * np.ones_like(k_on)
        variables.k_on = k_on
        variables.k_on_shape = k_on_shape
        variables.k_on_scale = k_on_scale

        # Set up k_off
        if k_off is None:
            if num_micro == 2:
                k_off = 1000/(dt*num_frames) * np.array([1, 0])
            elif num_micro == 3:
                k_off = 1000/(dt*num_frames) * np.array([1, 0, 0])
            elif num_micro == 4:
                k_off = 1000/(dt*num_frames) * np.array([1, num_frames/1000, 0, 0])
        k_off = k_off * np.ones(num_micro)
        if k_off_scale is None:
            k_off_scale = k_off / k_off_shape
        k_off_shape = k_off_shape * np.ones_like(k_off)
        variables.k_off = k_off
        variables.k_off_scale = k_off_scale
        variables.k_off_shape = k_off_shape

        # Set up k_photo
        if k_photo is None:
            if num_micro == 2:
                k_photo = 1000/(dt*num_frames) * np.array([
                    [-1, 1],  # Bright to Bleached
                    [0, 0],   # Irreversible Bleaching 
                ])
            elif num_micro == 3:
                k_photo = 1000/(dt*num_frames) * np.array([
                    [-1, .1, .9],  # Bright to Dark or Bleached
                    [1, -1, 0],    # Dark to Bright
                    [0, 0, 0],     # Irreversible Bleaching
                ])
            elif num_micro == 4:
                k_photo = 1000/(dt*num_frames) * np.array([
                    [-1, 0, .1, .9],  # Bright to Dark and Bleached
                    [0, 0, 0, 0],     # No nonspecific to bright
                    [1, 0, -1, 0],    # Dark to Bright
                    [0, 0, 0, 0],     # Irreversible Bleaching
                ])
        k_photo_shape = k_photo_shape * np.ones_like(k_photo)
        if k_photo_scale is None:
            k_photo_scale = k_photo / k_photo_shape
        variables.k_photo = k_photo
        variables.k_photo_scale = k_photo_scale
        variables.k_photo_shape = k_photo_shape

        # Set up statespace variables
        if bright_ids is None:
            if num_micro == 2:
                bright_ids = [0]
            elif num_micro == 3:
                bright_ids = [0]
            elif num_micro == 4:
                bright_ids = [0, 1]
        if microstate_names is None:
            if num_micro == 2:
                microstate_names = ['Bright', 'Unbound/Bleached']
            elif num_micro == 3:
                microstate_names = ['Bright', 'Dark', 'Unbound/Bleached']
            elif num_micro == 4:
                microstate_names = ['Bright', 'Nonspecific', 'Dark', 'Unbound/Bleached']
        if num_max == None:
            num_max = int(np.max(data) * k_photo.shape[1])
        partitions = RateInference.calculate_partitions(num_max, num_micro)
        num_macro = partitions.shape[0]
        lhood_dict = tuple([np.where(np.sum(partitions[:, bright_ids], axis=1) == i)[0] for i in range(num_max+1)])
        variables.microstate_names = microstate_names
        variables.num_max = num_max
        variables.partitions = partitions
        variables.num_macro = num_macro
        variables.lhood_dict = lhood_dict
        variables.bright_ids = bright_ids

        # Set up pi0
        if pi0 is None:
            pi0 = np.zeros(num_macro)
            for k in range(num_macro):
                pi0[k] = np.prod(.5 ** partitions[k, :-1])
            pi0 /= np.sum(pi0)
        variables.pi0 = pi0

        # Set up P
        P = -np.inf
        variables.P = P

        # Return variables
        return variables
    
    # Define calculate_posterior
    @staticmethod
    def calculate_posterior(data, variables, **kwargs):

        # Set up variables
        if len(kwargs) > 0:
            variables = copy.deepcopy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)
        P = variables.P
        pi0 = variables.pi0
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        dt = variables.dt
        settings = variables.settings
        laserpower = variables.laserpower
        concentration = variables.concentration
        partitions = variables.partitions
        lhood_dict = variables.lhood_dict
        num_rois = variables.num_rois
        num_frames = variables.num_frames
        num_settings = variables.num_settings
        num_micro = variables.num_micro
        num_macro = variables.num_macro
        num_max = variables.num_max
        proposal_shape = variables.proposal_shape
        parallelize = variables.parallelize

        # Calculate prior
        P = 0
        for k in range(num_micro):
            # k_on prior
            if k_on_scale[k] > 0:
                P += stats.gamma.logpdf(k_on[k], k_on_shape[k], scale=k_on_scale[k])
            # k_off prior
            if k_off_scale[k] > 0:
                P += stats.gamma.logpdf(k_off[k], k_off_shape[k], scale=k_off_scale[k])
            # k_photo prior
            ids = k_photo[k, :] > 0
            if np.any(ids) and np.any(k_photo_scale[k, ids]) > 0:
                P += np.sum(stats.gamma.logpdf(k_photo[k, ids], k_photo_shape[k, ids], scale=k_photo_scale[k, ids]))

        # Calculate likelihood
        for k, (c, l) in enumerate(settings):

            # Get transition matrix
            pi = RateInference.micro2macro(
                k_on, k_off, k_photo, dt, c, l, partitions=partitions
            )

            # Define likelihood function
            def lhood(r):
                output = 0
                if r < data.shape[0]:
                    output = RateInference.likelihood(data[r, :], pi, lhood_dict)
                return output
            
            # Loop over rois
            ids = np.where((laserpower == l) & (concentration == c))[0]
            if parallelize:
                # Parallelize
                n_tries = 0
                while n_tries < 10:
                    try:
                        lhoods = Parallel(n_jobs=4)(delayed(lhood)(r) for r in ids)
                        break
                    except:
                        print(f"Failed to parallelize {n_tries}/10")
                        n_tries += 1
                P += np.sum(lhoods)
            else:
                # Serial
                for r in ids:
                    P += lhood(r)

        # Return P
        if not np.isfinite(P):
            P = -np.inf
        return P
    
    # Define sample_rates
    @staticmethod
    def sample_rates(data, variables, verbose=False):

        # Extract variables
        dt = variables.dt
        P = variables.P
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        laserpower = variables.laserpower
        concentration = variables.concentration
        settings = variables.settings
        num_rois = variables.num_rois
        num_micro = variables.num_micro
        num_frames = variables.num_frames
        num_settings = variables.num_settings
        proposal_shape = variables.proposal_shape

        # Sample k_on
        for k in range(num_micro):
            if k_on_scale[k] <= 0:
                continue
            # Get old
            k_on_old = k_on
            # Propose new
            k_on_new = k_on_old.copy()
            k_on_new[k] = stats.gamma.rvs(proposal_shape, scale=k_on_old[k]/proposal_shape)
            # Calculate acceptance probability
            P_old = RateInference.calculate_posterior(data, variables, k_on=k_on_old)
            P_new = RateInference.calculate_posterior(data, variables, k_on=k_on_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(k_on_old[k], proposal_shape, scale=k_on_new[k]/proposal_shape)
                - stats.gamma.logpdf(k_on_new[k], proposal_shape, scale=k_on_old[k]/proposal_shape)
            )
            # Accept or reject
            if acc_prob > np.log(np.random.rand()):
                k_on = k_on_new
                P = P_new
            if verbose:
                print(f"%", end='')

        # Sample k_off
        for k in range(num_micro):
            if k_off_scale[k] <= 0:
                continue
            # Get old
            k_off_old = k_off
            # Propose new
            k_off_new = k_off_old.copy()
            k_off_new[k] = stats.gamma.rvs(proposal_shape, scale=k_off_old[k]/proposal_shape)
            # Calculate acceptance probability
            P_old = RateInference.calculate_posterior(data, variables, k_off=k_off_old)
            P_new = RateInference.calculate_posterior(data, variables, k_off=k_off_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(k_off_old[k], proposal_shape, scale=k_off_new[k]/proposal_shape)
                - stats.gamma.logpdf(k_off_new[k], proposal_shape, scale=k_off_old[k]/proposal_shape)
            )
            # Accept or reject
            if acc_prob > np.log(np.random.rand()):
                k_off = k_off_new
                P = P_new
            if verbose:
                print(f"%", end='')

        # Sample k_photo
        for i in range(k_photo.shape[0]):
            for j in range(k_photo.shape[1]):
                if k_photo[i, j] <= 0 or k_photo_scale[i, j] <= 0:
                    continue
                # Get old
                k_photo_old = k_photo
                # Propose new
                k_photo_new = k_photo_old.copy()
                k_photo_new[i, j] = stats.gamma.rvs(proposal_shape, scale=k_photo_old[i, j]/proposal_shape)
                k_photo_new[i, i] = -(np.sum(k_photo_new[i, :]) - k_photo_new[i, i])
                # Calculate acceptance probability
                P_old = RateInference.calculate_posterior(data, variables, k_photo=k_photo_old)
                P_new = RateInference.calculate_posterior(data, variables, k_photo=k_photo_new)
                acc_prob = (
                    P_new - P_old
                    + stats.gamma.logpdf(k_photo_old[i, j], proposal_shape, scale=k_photo_new[i, j]/proposal_shape)
                    - stats.gamma.logpdf(k_photo_new[i, j], proposal_shape, scale=k_photo_old[i, j]/proposal_shape)
                )
                # Accept or reject
                if acc_prob > np.log(np.random.rand()):
                    k_photo = k_photo_new
                    P = P_new
                if verbose:
                    print(f"%", end='')

        # Update variables
        variables.P = P
        variables.k_on = k_on
        variables.k_off = k_off
        variables.k_photo = k_photo

        # Return variables
        return variables

    # Define plot_rates
    @staticmethod
    def plot_rates(MAP, Samples, groundtruth=None):
        return plot_rates(MAP, Samples, groundtruth=groundtruth)
    
    # Define plot states
    @staticmethod
    def plot_states(data, variables, r=None, **kwargs):

        # Extract variables
        dt = variables.dt
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        settings = variables.settings
        num_rois = variables.num_rois
        num_micro = variables.num_micro
        num_frames = variables.num_frames
        partitions = variables.partitions
        concentration = variables.concentration
        laserpower = variables.laserpower
        lhood_dict = variables.lhood_dict
        microstate_names = variables.microstate_names

        # Get roi
        if r is None:
            r = np.random.randint(0, num_rois)

        # Get state trajectory
        c = concentration[r]
        l = laserpower[r]
        pi = RateInference.micro2macro(k_on, k_off, k_photo, dt, c, l, partitions)
        states = RateInference.viterbi(data[r, :], pi, lhood_dict)

        # Create trace
        trace = np.zeros((num_micro, num_frames), dtype=int)
        for n in range(num_frames):
            trace[:, n] = partitions[states[n], :]

        # Initialize Plot
        fig = plt.gcf()
        plt.clf()
        plt.ion()
        plt.show()
        ax = np.empty((num_micro, 1), dtype=object)
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j] = fig.add_subplot(ax.shape[0], ax.shape[1], i*ax.shape[1]+j+1)

        # Plot data
        ax[0, 0].set_title(f'Data from ROI {r}')
        ax[0, 0].plot(data[r, :], color='g', label='Data')

        # Plot states
        for k in range(num_micro-1):
            ax[k+1, 0].set_title(microstate_names[k])
            ax[k+1, 0].plot(trace[k, :], color='b', label=f'State {k}')

        # Finalize plot
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].set_ylabel('Number bound')
                ax[i, j].legend()
                if i == ax.shape[0]-1:
                    ax[i, j].set_xlabel('Frame')
        plt.tight_layout()
        plt.pause(0.1)
        return    
        
    # Define analze
    @staticmethod
    def analyze(data, parameters=None, num_iterations=1000, saveevery=2, savepath=None, plot=False, verbose=True, **kwargs):

        # Initialize variables
        if verbose:
            print("Initializing variables...")
        variables = RateInference.initialize_variables(data, parameters, **kwargs)
        MAPvariables = copy.deepcopy(variables)
        Samples = {
            "P": np.zeros((num_iterations//saveevery, *np.shape(variables.P))),
            "k_on": np.zeros((num_iterations//saveevery, *np.shape(variables.k_on))),
            "k_off": np.zeros((num_iterations//saveevery, *np.shape(variables.k_off))),
            "k_photo": np.zeros((num_iterations//saveevery, *np.shape(variables.k_photo))),
        }
        if verbose:
            print(f"Initialized variables")
            for key, value in sorted(variables.__dict__.items()):
                typestr = f"{type(value)}"
                if np.isscalar(value):
                    valstr = f"= {value}"
                elif isinstance(value, np.ndarray): # np.array is not a type, instead use np.ndarray
                    valstr = f"shape {np.shape(value)}"
                elif hasattr(value, '__len__'):
                    valstr = f"len {len(value)}"
                outstr = f"--{key} : {typestr} {valstr}"[:80]
                print(outstr)

        # Gibbs sampling
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration+1}/{num_iterations} [", end='')
            t = time.time()
                
            # Sample variables
            variables = RateInference.sample_rates(data, variables, verbose=verbose)

            # Save samples
            if variables.P >= MAPvariables.P:
                MAPvariables = copy.deepcopy(variables)
            if iteration % saveevery == 0:
                Samples["P"][iteration//saveevery] = variables.P
                Samples["k_on"][iteration//saveevery] = variables.k_on.copy()
                Samples["k_off"][iteration//saveevery] = variables.k_off.copy()
                Samples["k_photo"][iteration//saveevery] = variables.k_photo.copy()

                if savepath is not None:
                    output = {
                        **parameters,
                        "variables": variables.__dict__,
                        "samples": Samples,
                        "datapath": "none",
                    }
                    with h5py.File(savepath, "w") as h5:
                        dict_to_h5(h5, output)

            # Plot
            if plot and iteration % 10 == 0:
                RateInference.plot_rates(MAPvariables, Samples)
                plt.pause(.1)
                # RateInference.plot_states(data, MAPvariables)
                # plt.pause(.1)

            # Print
            if verbose:
                print(f"] ({time.time()-t:.2f} s) P = {variables.P:.2f}")

        # Return MAP variables
        return MAPvariables, Samples


