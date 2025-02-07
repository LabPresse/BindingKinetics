
# Import standard modules
import os
import sys
import h5py
import copy
import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace
from src.ffbs import FFBS
from src.PlotData import plot_data



# Declare class
class StateInference:

    # Declare parameters
    PARAMETERS = {
        # Variables
        "s": None,
        "pi": None,
        "mu_flor": None,
        "mu_back": None,
        "sigma_flor": None,
        "sigma_back": None,
        "P": None,
        # Priors
        "pi_gamma": .1,
        "pi_conc": None,
        "mu_flor_shape": 2,
        "mu_flor_scale": None,
        "mu_back_shape": 2,
        "mu_back_scale": None,
        "sigma_flor_shape": 2,
        "sigma_flor_scale": None,
        "sigma_back_shape": 2,
        "sigma_back_scale": None,
        # Numbers
        "num_states": 10,
        "num_frames": None,
        # Sampling
        "mu_flor_proposal_shape": 10,
        "mu_back_proposal_shape": 10,
        "sigma_flor_proposal_shape": 10,
        "sigma_back_proposal_shape": 10,
    }
    
    @staticmethod
    def calculate_likelihood(data, variables, **kwargs):

        # Set up variables
        if len(kwargs) > 0:
            variables = copy.copy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)

        # Extract variables
        s = variables.s
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        sigma_flor = variables.sigma_flor
        sigma_back = variables.sigma_back
        num_states = variables.num_states
        
        # Calculate likelihood
        likelihood = 0
        for k in range(num_states):
            if np.any(s==k):
                mu = k * mu_flor + mu_back
                sigma = k * sigma_flor + sigma_back
                likelihood += np.sum(stats.norm.logpdf(data[s==k], loc=mu, scale=sigma))

        # Return likelihood
        return likelihood

    @staticmethod
    def calculate_posterior(data, variables, **kwargs):

        # Set up variables
        if len(kwargs) > 0:
            variables = copy.copy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)
        
        # Get variables
        s = variables.s
        pi = variables.pi
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        sigma_flor = variables.sigma_flor
        sigma_back = variables.sigma_back
        pi_conc = variables.pi_conc
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        sigma_flor_shape = variables.sigma_flor_shape
        sigma_flor_scale = variables.sigma_flor_scale
        sigma_back_shape = variables.sigma_back_shape
        sigma_back_scale = variables.sigma_back_scale
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Calculate prior
        prior = (
            stats.gamma.logpdf(mu_flor, a=mu_flor_shape, scale=mu_flor_scale)
            + stats.gamma.logpdf(mu_back, a=mu_back_shape, scale=mu_back_scale)
            + stats.gamma.logpdf(sigma_flor, a=sigma_flor_shape, scale=sigma_flor_scale)
            + stats.gamma.logpdf(sigma_back, a=sigma_back_shape, scale=sigma_back_scale)
        )
        for k in range(num_states + 1):
            ids = pi_conc[k, :] > 0
            prior += stats.dirichlet.logpdf(pi[k, ids], pi_conc[k, ids])

        # Calculate likelihood
        likelihood = StateInference.calculate_likelihood(data, variables)

        # Calculate dynamics
        dynamics = 0
        s_old = -1
        for n in range(num_frames):
            s_new = s[n]
            dynamics += np.log(pi[s_old, s_new])
            s_old = s_new

        # Calculate posterior
        P = prior + likelihood + dynamics

        return P

    @staticmethod
    def initialize_variables(data, parameters=None, **kwargs):
        
        # Set up parameters
        if parameters is None:
            parameters = {}
        parameters = copy.deepcopy(parameters)
        parameters = {**copy.deepcopy(StateInference.PARAMETERS), **parameters, **kwargs}

        # Set up variables
        variables = SimpleNamespace(**parameters)
        P = variables.P
        s = variables.s 
        pi = variables.pi
        pi_gamma = variables.pi_gamma
        pi_conc = variables.pi_conc
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        sigma_flor = variables.sigma_flor
        sigma_flor_shape = variables.sigma_flor_shape
        sigma_flor_scale = variables.sigma_flor_scale
        sigma_back = variables.sigma_back
        sigma_back_shape = variables.sigma_back_shape
        sigma_back_scale = variables.sigma_back_scale
        num_states = variables.num_states
        num_frames = variables.num_frames
        mu_flor_proposal_shape = variables.mu_flor_proposal_shape
        mu_back_proposal_shape = variables.mu_back_proposal_shape

        # Initialize constants
        num_frames = data.shape[0]
        variables.num_frames = num_frames

        # Initialize transition probabilities
        if pi_conc is None:
            pi_conc = np.zeros((num_states+1, num_states))
            pi_conc[-1, :] = pi_gamma ** (np.arange(num_states))
            pi_conc[:-1, :] += (
                10 * np.eye(num_states)
                + 1*np.eye(num_states, k=1)
                + 2*np.eye(num_states, k=-1)
            )
            for k in range(num_states + 1):
                pi_conc[k, :] /= np.sum(pi_conc[k, :])
        if pi is None:
            pi = pi_conc.copy()
            for k in range(num_states + 1):
                pi[k, :] /= np.sum(pi[k, :])
        variables.pi = pi
        variables.pi_conc = pi_conc

        # Initialize states
        s = np.zeros(num_frames, dtype=int)
        variables.s = s

        # Initialize background
        if (mu_back_scale is None) and (data is not None):
            mu_back_scale = np.mean(data) / mu_back_shape
        if mu_back is None:
            mu_back = mu_back_shape * mu_back_scale
        variables.mu_back = mu_back
        variables.mu_back_scale = mu_back_scale

        # Initialize fluorophore brightness
        if (mu_flor_scale is None) and (data is not None):
            mu_flor_scale = np.std(data) / mu_flor_shape
        if mu_flor is None:
            mu_flor = mu_flor_shape * mu_flor_scale
        variables.mu_flor = mu_flor
        variables.mu_flor_scale = mu_flor_scale

        # Initialize background noise
        if (sigma_back_scale is None) and (data is not None):
            sigma_back_scale = np.std(data) / sigma_back_shape
        if sigma_back is None:
            sigma_back = sigma_back_shape * sigma_back_scale
        variables.sigma_back = sigma_back
        variables.sigma_back_scale = sigma_back_scale

        # Initialize fluorophore noise
        if (sigma_flor_scale is None) and (data is not None):
            sigma_flor_scale = np.std(data) / sigma_flor_shape / 100
        if sigma_flor is None:
            sigma_flor = sigma_flor_shape * sigma_flor_scale
        variables.sigma_flor = sigma_flor
        variables.sigma_flor_scale = sigma_flor_scale

        # Initialze probability
        P = -np.inf
        variables.P = P
            
        # Return variables
        return variables

    @staticmethod
    def sample_states(data, variables):
        
        # Extract variables
        s = variables.s
        pi = variables.pi
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        sigma_flor = variables.sigma_flor
        sigma_back = variables.sigma_back
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Set up log likelihood matrix
        lhood = np.zeros((num_states, num_frames))
        for k in range(num_states):
            mu = k * mu_flor + mu_back
            sigma = k * sigma_flor + sigma_back
            lhood[k, :] = stats.norm.logpdf(data, mu, sigma)

        # Softmax for numerical stability
        lhood = np.exp(lhood - np.max(lhood, axis=0))

        # Sample states using FFBS
        s[:] = FFBS(lhood, pi)

        # Update variables
        variables.s = s

        # Return variables
        return variables

    @staticmethod
    def sample_transitions(data, variables):
        
        # Extract variables
        s = variables.s
        pi = variables.pi
        pi_conc = variables.pi_conc
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Get counts
        counts = np.zeros((num_states+1, num_states))
        s_old = -1
        for n in range(num_frames):
            s_new = s[n]
            counts[s_old, s_new] += 1
            s_old = s_new

        # Sample transition probabilities
        for k in range(num_states):  # Not sampling last row
            ids = pi_conc[k, :] > 0
            pi[k, ids] = stats.dirichlet.rvs(counts[k, ids] + pi_conc[k, ids])

        # Return variables
        return variables

    @staticmethod
    def sample_brightness(data, variables):

        # Extract variables
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        mu_flor_proposal_shape = variables.mu_flor_proposal_shape
        mu_back_proposal_shape = variables.mu_back_proposal_shape

        # Define conditional probability
        def prob(mu_flor_, mu_back_):
            P = (
                StateInference.calculate_likelihood(data, variables, mu_flor=mu_flor_, mu_back=mu_back_)
                + stats.gamma.logpdf(mu_flor_, a=mu_flor_shape, scale=mu_flor_scale)
                + stats.gamma.logpdf(mu_back_, a=mu_back_shape, scale=mu_back_scale)
            )
            return P

        # Sample brightnesses
        for _ in range(10):

            # Sample background
            a = mu_back_proposal_shape
            mu_back_old = copy.deepcopy(mu_back)
            mu_back_new = stats.gamma.rvs(a=a, scale=mu_back_old/a)
            P_old = prob(mu_flor, mu_back_old)
            P_new = prob(mu_flor, mu_back_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(mu_back_old, a=a, scale=mu_back_new/a)
                - stats.gamma.logpdf(mu_back_new, a=a, scale=mu_back_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                mu_back = mu_back_new
            
            # Sample fluorophore brightness
            a = mu_flor_proposal_shape
            mu_flor_old = copy.deepcopy(mu_flor)
            mu_flor_new = stats.gamma.rvs(a=a, scale=mu_flor_old/a)
            P_old = prob(mu_flor_old, mu_back)
            P_new = prob(mu_flor_new, mu_back)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(mu_flor_old, a=a, scale=mu_flor_new/a)
                - stats.gamma.logpdf(mu_flor_new, a=a, scale=mu_flor_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                mu_flor = mu_flor_new

        # Update variables
        variables.mu_flor = mu_flor
        variables.mu_back = mu_back

        # Return variables
        return variables
    
    @staticmethod
    def sample_noise(data, variables):

        # Extract variables
        sigma_flor = variables.sigma_flor
        sigma_flor_shape = variables.sigma_flor_shape
        sigma_flor_scale = variables.sigma_flor_scale
        sigma_back = variables.sigma_back
        sigma_back_shape = variables.sigma_back_shape
        sigma_back_scale = variables.sigma_back_scale
        sigma_flor_proposal_shape = variables.sigma_flor_proposal_shape
        sigma_back_proposal_shape = variables.sigma_back_proposal_shape

        # Define conditional probability
        def prob(sigma_flor_, sigma_back_):
            P = (
                StateInference.calculate_likelihood(
                    data, variables, sigma_flor=sigma_flor_, sigma_back=sigma_back_
                )
                + stats.gamma.logpdf(sigma_flor_, a=sigma_flor_shape, scale=sigma_flor_scale)
                + stats.gamma.logpdf(sigma_back_, a=sigma_back_shape, scale=sigma_back_scale)
            )
            return P
        
        # Sample noise
        for _ in range(10):

            # Sample background
            a = sigma_back_proposal_shape
            sigma_back_old = copy.deepcopy(sigma_back)
            sigma_back_new = stats.gamma.rvs(a=a, scale=sigma_back_old/a)
            P_old = prob(sigma_flor, sigma_back_old)
            P_new = prob(sigma_flor, sigma_back_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(sigma_back_old, a=a, scale=sigma_back_new/a)
                - stats.gamma.logpdf(sigma_back_new, a=a, scale=sigma_back_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                sigma_back = sigma_back_new

            # Sample fluorophore brightness
            a = sigma_flor_proposal_shape
            sigma_flor_old = copy.deepcopy(sigma_flor)
            sigma_flor_new = stats.gamma.rvs(a=a, scale=sigma_flor_old/a)
            P_old = prob(sigma_flor_old, sigma_back)
            P_new = prob(sigma_flor_new, sigma_back)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(sigma_flor_old, a=a, scale=sigma_flor_new/a)
                - stats.gamma.logpdf(sigma_flor_new, a=a, scale=sigma_flor_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                sigma_flor = sigma_flor_new

        # Update variables
        variables.sigma_flor = sigma_flor
        variables.sigma_back = sigma_back

        # Return variables
        return variables

    @staticmethod
    def sample_states_and_brightness(data, variables):

        # Extract variables
        pi_conc = variables.pi_conc
        num_states = variables.num_states

        # Sample sampling style
        style = ["exp", "*2", "/2"][np.random.randint(3)]

        # Propose new brightnesses and transition probabilities
        variables_old = variables
        mu_flor_old = variables_old.mu_flor
        pi_old = variables_old.pi
        # Propose new brightness
        if style == "exp":
            mu_flor_new = stats.expon.rvs(scale=mu_flor_old)
        elif style == "*2":
            mu_flor_new = mu_flor_old * 2
        elif style == "/2":
            mu_flor_new = mu_flor_old / 2
        # Propose new transition probabilities
        pi_new = pi_old.copy()
        for k in range(num_states):
            ids = pi_conc[k, :] > 0
            pi_new[k, ids] = stats.dirichlet.rvs(pi_conc[k, ids])
        # Sample new states
        variables_new = copy.deepcopy(variables)
        variables_new.mu_flor = mu_flor_new
        variables_new.pi = pi_new
        variables_new = StateInference.sample_states(data, variables_new)

        # Calculate acceptance probability
        P_old = StateInference.calculate_posterior(data, variables_old)
        P_new = StateInference.calculate_posterior(data, variables_new)
        acc_prob = P_new - P_old
        for k in range(num_states + 1):
            ids = pi_conc[k, :] > 0
            acc_prob += (
                stats.dirichlet.logpdf(pi_new[k, ids], pi_conc[k, ids])
                - stats.dirichlet.logpdf(pi_old[k, ids], pi_conc[k, ids])
            )
        if style == "exp":
            acc_prob += (
                + stats.expon.logpdf(mu_flor_old, scale=mu_flor_new)
                - stats.expon.logpdf(mu_flor_new, scale=mu_flor_old)
            )
        if acc_prob > np.log(np.random.rand()):
            variables = variables_new
        
        # Return variables
        return variables
    
    @staticmethod
    def sample_states_and_background(data, variables):

        # Extract variables
        pi_conc = variables.pi_conc
        num_states = variables.num_states

        # Sample sampling style
        style = ["exp", "+1", "-1"][np.random.randint(3)]

        # Propose new brightnesses and transition probabilities
        variables_old = variables
        mu_back_old = variables_old.mu_back
        pi_old = variables_old.pi
        # Propose new background brightness
        if style == "exp":
            mu_back_new = stats.expon.rvs(scale=mu_back_old)
        elif style == "+1":
            mu_back_new = mu_back_old + variables.mu_flor
        elif style == "-1":
            mu_back_new = mu_back_old - variables.mu_flor
            if mu_back_new < 0:
                return variables
        # Propose new transition probabilities
        pi_new = pi_old.copy()
        for k in range(variables.num_states):
            ids = variables.pi_conc[k, :] > 0
            pi_new[k, ids] = stats.dirichlet.rvs(variables.pi_conc[k, ids])
        # Sample new states
        variables_new = copy.deepcopy(variables)
        variables_new.pi = pi_new
        variables_new.mu_back = mu_back_new
        variables_new = StateInference.sample_states(data, variables_new)

        # Calculate acceptance probability
        P_old = StateInference.calculate_posterior(data, variables_old)
        P_new = StateInference.calculate_posterior(data, variables_new)
        acc_prob = P_new - P_old
        for k in range(num_states + 1):
            ids = pi_conc[k, :] > 0
            acc_prob += (
                stats.dirichlet.logpdf(pi_new[k, ids], pi_conc[k, ids])
                - stats.dirichlet.logpdf(pi_old[k, ids], pi_conc[k, ids])
            )
        if style == "exp":
            acc_prob += (
                + stats.expon.logpdf(mu_back_old, scale=mu_back_new)
                - stats.expon.logpdf(mu_back_new, scale=mu_back_old)
            )
        if acc_prob > np.log(np.random.rand()):
            variables = variables_new
        
        # Return variables
        return variables
    
    @staticmethod
    def plot_data(data, variables=None, groundtruth=None):
        plot_data(data, variables, groundtruth)
        return

    @staticmethod
    def analyze(data, parameters=None, num_iterations=500, plot=False, verbose=True, **kwargs):

        # Initialize variables
        variables = StateInference.initialize_variables(data, parameters, **kwargs)
        MAPvariables = copy.deepcopy(variables)
        if verbose:
            print(f"Initialized variables")
            for key, value in sorted(variables.__dict__.items()):
                typestr = f"{type(value)}"
                if np.isscalar(value):
                    valstr = f"= {value}"
                else:
                    valstr = ""
                outstr = f"--{key} : {typestr} {valstr}"[:80]
                print(outstr)

        # Gibbs sampling
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration+1}/{num_iterations}", end='')
            t = time.time()
                
            # Sample states
            variables = StateInference.sample_noise(data, variables)
            variables = StateInference.sample_brightness(data, variables)
            variables = StateInference.sample_transitions(data, variables)
            variables = StateInference.sample_states(data, variables)
            if np.random.rand() < .25:
                variables = StateInference.sample_states_and_brightness(data, variables)
            if np.random.rand() < .25:
                variables = StateInference.sample_states_and_background(data, variables)

            # Update MAP
            variables.P = StateInference.calculate_posterior(data, variables)
            if variables.P >= MAPvariables.P:
                MAPvariables = copy.deepcopy(variables)

            # Plot
            if plot and iteration % 10 == 0:
                StateInference.plot_data(data, variables)
                plt.pause(.1)

            # Print
            if verbose:
                print(f" ({time.time()-t:.2f} s) P = {variables.P:.2f}")

        # Return MAP variables
        return MAPvariables


