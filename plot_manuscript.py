
# Import libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.dict2hdf5 import h5_to_dict
from src.load_data import load_data

# Load configuration
from config import *


### Define load functions ###

# Load results
def load_results(datapath, statespath, ratespath, merge=None, root=None):

    # Check if root is specified
    if root is None:
        root = ''
    
    # Load data, states, and rates
    data, metadata = load_data(root+datapath, merge=merge)
    with h5py.File(root+statespath, "r") as f:
        statevars = h5_to_dict(f)
        states = statevars["states"]
        mu_flor = statevars["mu_flor"]
        mu_back = statevars["mu_back"]
    with h5py.File(root+ratespath, "r") as f:
        ratesvars = h5_to_dict(f)

    # Filter data
    dataraw = data.copy()
    statemask = statevars["mask"]
    ratemask = ratesvars["statesmask"]
    data = data[statemask, :]
    data = data[ratemask, :]
    states = states[ratemask, :]
    mu_flor = mu_flor[ratemask]
    mu_back = mu_back[ratemask]

    # Calculate trace
    mu_back = mu_back + np.min(states, axis=1)*mu_flor  # Shift baseline
    states = states - np.min(states, axis=1)[:, None]  # Shift states
    trace = mu_flor[:, None]*states
    trace = (
        trace 
        - np.mean(trace, axis=1)[:, None]  
        + np.mean(data, axis=1)[:, None]
    )

    # Return
    return data, trace, ratesvars


### Define plot functions ###

# Lifetimes
def plot_lifetimes(ax, data, groundtruth=None, bins=10):

    # Take out burn-in
    last = np.where(data != 0)[0][-1]
    data = data[last//2:last]
    print(data.mean() * 1e6, data.std() * 1e6)

    # Rescale data to mHz
    data = data*1e6
    if groundtruth is not None:
        groundtruth = groundtruth*1e6

    # Histogram data
    ax.hist(data, bins=bins, density=True, color='b', alpha=0.5, label='Samples')
    if groundtruth is not None:
        ax.axvline(groundtruth, color='r', label='Ground truth')
    
    # Return
    return ax

# Data
def plot_data(ax, data, states=None):

    # Plot data
    times = .124*np.arange(data.shape[0])
    ax.plot(times, data, linewidth=2, color='g', alpha=0.5, label='Data')
    if states is not None:
        ax.plot(times, states, ':', linewidth=1, color='b', label='States')
    
    # Return
    return ax


### Set up figure ###

# Results
def figure_results(filelist):

    # Set up files and paths
    root = os.join(PATH_DATA, "/Binding/")

    # Create a figure
    fig = plt.figure(figsize=(10, 6))
    plt.ion()
    plt.show()

    # Add axes
    gs = gridspec.GridSpec(6, 4)
    ax = np.empty((3, 4), dtype=object)
    for i in range(3):
        if i == 0:
            sharex = [None, None, None, None]
        else:
            sharex = [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3]]
        ax[i, 0] = fig.add_subplot(gs[2*i, 0:2], sharex=sharex[0])
        ax[i, 1] = fig.add_subplot(gs[2*i+1, 0:2], sharex=sharex[1])
        ax[i, 2] = fig.add_subplot(gs[2*i:2*(i+1), 2], sharex=sharex[2])
        ax[i, 3] = fig.add_subplot(gs[2*i:2*(i+1), 3], sharex=sharex[3])

    # Loop over files
    for i, (datafile, statesfile, ratesfile, merge) in enumerate(filelist):

        # Load results
        data, trace, rates = load_results(
            datapath=datafile,
            statespath=statesfile,
            ratespath=ratesfile,
            merge=merge,
            root=root,
        )

        # Plot data
        rois = np.random.permutation(np.arange(data.shape[0]))[:2]
        plot_data(ax[i, 0], data[rois[0], :], trace[rois[0], :])
        plot_data(ax[i, 1], data[rois[1], :], trace[rois[1], :])

        # Plot lifetimes
        plot_lifetimes(ax[i, 2], rates['samples']["k_on"][:, 0])
        plot_lifetimes(ax[i, 3], rates['samples']["k_off"][:, 0])

    # Remove axes from traces
    for i in range(ax.shape[0]):
        for j in range(2):
            ax[i, j].set_yticks([])
            if not (i == 2 and j == 1):
                plt.setp(ax[i, j].get_xticklabels(), visible=False)
    
    # Remove axes from lifetimes
    for i in range(ax.shape[0]):
        for j in range(2, 4):
            ax[i, j].set_yticks([])
            if i == ax.shape[0]-1:
                ax[i, j].tick_params(axis='x', rotation=45)
            else:
                plt.setp(ax[i, j].get_xticklabels(), visible=False)

    # Set labels
    ax[0, 0].set_title('Sample Data')
    ax[0, 2].set_title('K on')
    ax[0, 3].set_title('K off')
    ax[-1, 1].set_xlabel("Time (s)")
    ax[-1, 2].set_xlabel("K on (mHz/nM)")
    ax[-1, 3].set_xlabel("K off (mHz)")
    # Rotate binding site labels
    ax[0, 0].set_ylabel("One\nBinding\nSite", rotation=0, labelpad=20)
    ax[1, 0].set_ylabel("Two\nBinding\nSites", rotation=0, labelpad=20)
    ax[2, 0].set_ylabel("Five\nBinding\nSites", rotation=0, labelpad=20)

    # Set legend
    ax[0, 0].legend(loc='upper right')
    ax[0, -1].legend(loc='upper right')

    # Return
    return fig, ax

# Simulation
def figure_simulation(test='kon'):


    # Set up files
    if test == 'kon':
        sim_files = [
            "simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2",
            "simulated_kon=2e-6_koff=2e-5_kphoto=1e-7_Nstates=2",
            "simulated_kon=5e-6_koff=2e-5_kphoto=1e-7_Nstates=2",
        ]
        sim_files = [(f+".h5", f+"_STATES.h5", f+"_minstep=5_nummicro=2_RATES.h5", None) for f in sim_files]
    elif test == 'koff':
        sim_files = [
            "simulated_kon=1e-6_koff=1e-5_kphoto=1e-7_Nstates=2",
            "simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2",
            "simulated_kon=1e-6_koff=5e-5_kphoto=1e-7_Nstates=2",
        ]
        sim_files = [(f+".h5", f+"_STATES.h5", f+"_minstep=5_nummicro=2_RATES.h5", None) for f in sim_files]
    elif test == 'kphoto':
        sim_files = [
            "simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2",
            "simulated_kon=1e-6_koff=2e-5_kphoto=2e-7_Nstates=2",
            "simulated_kon=1e-6_koff=2e-5_kphoto=5e-7_Nstates=2",
        ]
        sim_files = [(f+".h5", f+"_STATES.h5", f+"_minstep=5_nummicro=2_RATES.h5", None) for f in sim_files]
    elif test == 'ndata':
        f = "simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2"
        sim_files = [
            (f+".h5", f+"_STATES.h5", f+"_minstep=5_nummicro=2_RATES.h5", None),
            (f+".h5", f+"_STATES.h5", f+"_downsample=2_minstep=5_nummicro=2_RATES.h5", None),
            (f+".h5", f+"_STATES.h5", f+"_downsample=5_minstep=5_nummicro=2_RATES.h5", None),
        ]


    # Initialize figure
    fig, ax = figure_results(sim_files)

    # Add ground truth
    for i in range(ax.shape[0]):
        if test == 'kon':
            k_on = 0.000001 * [1, 2, 5][i] * 1e6
            k_off = 0.000025 * 1e6
        elif test == 'koff':
            k_on = 0.000001 * 1e6
            k_off = 0.00001 * [1, 2, 5][i] * 1e6
        elif test == 'kphoto':
            k_on = 0.000001 * 1e6
            k_off = 0.000025 * 1e6
        elif test == 'ndata':
            k_on = 0.000001 * 1e6
            k_off = 0.000025 * 1e6
        ax[i, 2].axvline(k_on, color='r', label='Ground truth')
        ax[i, 3].axvline(k_off, color='r', label='Ground truth')

    # Set xlims of ax[:, [2, 3]] to start at 0
    for i in range(ax.shape[0]):
        for j in [2, 3]:
            ax[i, j].set_xlim([0, ax[i, j].get_xlim()[1]])

    # Set labels
    ax[0, 0].set_title('Sample Data')
    ax[0, 2].set_title('K bind')
    ax[0, 3].set_title('K un')
    ax[-1, 1].set_xlabel("Time (s)")
    ax[-1, 2].set_xlabel("K bind (mHz/nM)")
    ax[-1, 3].set_xlabel("K un (mHz)")
    # Rotate binding site labels
    if test == 'kon':
        ax[0, 0].set_ylabel(r"$K_{bind} = 1$ mHz/nM", rotation=0, labelpad=30)
        ax[1, 0].set_ylabel(r"$K_{bind} = 2$ mHz/nM", rotation=0, labelpad=30)
        ax[2, 0].set_ylabel(r"$K_{bind} = 5$ mHz/nM", rotation=0, labelpad=30)
    elif test == 'koff':
        ax[0, 0].set_ylabel(r"$K_{un} = 10$ mHz", rotation=0, labelpad=30)
        ax[1, 0].set_ylabel(r"$K_{un} = 20$ mHz", rotation=0, labelpad=30)
        ax[2, 0].set_ylabel(r"$K_{un} = 50$ mHz", rotation=0, labelpad=30)
    elif test == 'kphoto':
        ax[0, 0].set_ylabel(r"$K_{photo} = .1$ mHz", rotation=0, labelpad=30)
        ax[1, 0].set_ylabel(r"$K_{photo} = .2$ mHz", rotation=0, labelpad=30)
        ax[2, 0].set_ylabel(r"$K_{photo} = .5$ mHz", rotation=0, labelpad=30)
    elif test == 'ndata':
        ax[0, 0].set_ylabel(r"$N = 2000$", rotation=0, labelpad=30)
        ax[1, 0].set_ylabel(r"$N = 1000$", rotation=0, labelpad=30)
        ax[2, 0].set_ylabel(r"$N = 400$", rotation=0, labelpad=30)


    # Finalize
    plt.tight_layout()
    plt.pause(.1)

    # Return
    return fig, ax

# Experment
def figure_experiment():

    # Set up files and paths
    exp_files = [   
        # One Binding Site
        (
            "ST114_filtered.h5",
            "ST114_filtered_STATES.h5",
            "ST114_filtered_downsample=10_minstep=5_nummicro=2_RATES.h5",
            None,
        ),
        # Two Binding Sites
        (
            "ST128_filtered.h5",
            "ST128_filtered_STATES.h5",
            "ST128_filtered_downsample=10_minstep=5_nummicro=2_RATES.h5",
            None,
        ),
        # Four Binding Sites
        (
            "ST129_filtered.h5",
            "ST129_filtered_STATES.h5",
            "ST129_filtered_downsample=10_minstep=5_nummicro=2_RATES.h5",
            None,
        ),
    ]

    # Initialize figure
    fig, ax = figure_results(exp_files)

    # Set axis
    ax[0, 2].set_xlim([0, 10])
    ax[0, 3].set_xlim([0, 20])

    # Set labels
    ax[0, 0].set_title('Sample Data')
    ax[0, 2].set_title('K bind')
    ax[0, 3].set_title('K un')
    ax[-1, 1].set_xlabel("Time (s)")
    ax[-1, 2].set_xlabel("K bind (mHz/nM)")
    ax[-1, 3].set_xlabel("K un (mHz)")
    # Rotate binding site labels
    ax[0, 0].set_ylabel("One\nBinding\nSite", rotation=0, labelpad=20)
    ax[1, 0].set_ylabel("Two\nBinding\nSites", rotation=0, labelpad=20)
    ax[2, 0].set_ylabel("Five\nBinding\nSites", rotation=0, labelpad=20)

    # Set legend
    ax[0, 0].legend(loc='upper right')
    ax[0, -1].legend(loc='upper right')

    # Finalize
    plt.tight_layout()
    plt.pause(.1)

    # Return
    return fig, ax


# Main function
if __name__ == '__main__':

    ## Results

    # Simulation kon
    print("Simulation kon")
    figure_simulation(test='kon')
    # plt.savefig("outfiles/results_simulation_kon.png", dpi=300)

    # Simulation koff
    print("Simulation koff")
    figure_simulation(test='koff')
    # plt.savefig("outfiles/results_simulation_koff.png", dpi=300)

    # Simulation kphoto
    print("Simulation kphoto")
    figure_simulation(test='kphoto')
    # plt.savefig("outfiles/results_simulation_kphoto.png", dpi=300)

    # Simulation ndata
    print("Simulation ndata")
    figure_simulation(test='ndata')
    # plt.savefig("outfiles/results_simulation_ndata.png", dpi=300)

    # Experiment
    print("Experiment")
    figure_experiment()
    # plt.savefig("outfiles/results_experiment.png", dpi=300)

    # Done
    print("Done")

