
# Import libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from src.dict2hdf5 import h5_to_dict

# Load configuration
from config import *


# Define plot function
def plot(data, states, rois=0):

    # Select ROIs
    if np.isscalar(rois):
        rois = [rois]

    # Initialize figure
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    ax = np.empty(len(rois), dtype=object)
    for i, r in enumerate(rois):
        ax[i] = fig.add_subplot(len(rois), 1, i+1)

    # Plot rois
    colData = "r"
    colStates = "b"
    colLevels = "b"
    for i, r in enumerate(rois):
        if data is not None:
            # Plot data
            ax[i].plot(data["data"][r, :], c=colData, label="Data")
        if states is not None:
            # Extract variables
            mu_flor = states["mu_flor"][r]
            mu_back = states["mu_back"][r]
            s = states["states"][r, :]
            # Shift baseline
            mu_back = mu_back + np.min(s)*mu_flor
            s = s - np.min(s)
            # Plot trace
            ax[i].plot(mu_flor*s + mu_back, c=colStates, label="States")
            ax[i].axhline(mu_back, linestyle=":", c=colLevels, label="Baseline & Fluorophore")
            ax[i].axhline(mu_back+mu_flor, linestyle=":", c=colLevels)
    
    # Finalize plot
    ax[-1].legend()
    plt.tight_layout()
    plt.pause(.1)
    return


# Main script
if __name__ == "__main__":
    
    # Select file
    path = os.path.join(PATH_DATA, "Binding/")
    files = os.listdir(path)
    files = [
        # "SimulatedData_STATES.h5",
        # "SimulatedDataNoBlink_STATES.h5",
        "simulated_Kon10-6_Koff5-6_2states.h5",
        "simulated_Kon10-6_Koff12-5_2states.h5",
        "simulated_Kon10-6_Koff25-6_2states.h5",
    ]
    
    # Loop over files
    for file in files:

        # Open file
        with h5py.File(path + file, "r") as f:
            states_vars = h5_to_dict(f)
            states = states_vars["states"]

        # Open data
        datapath = os.path.join(
            PATH_DATA,
            states_vars["datapath"].decode().split("/Data/")[1]
        )
        with h5py.File(datapath, "r") as f:
            data_vars = h5_to_dict(f)
            data = data_vars["data"]

        # Filter data
        mask = states_vars["mask"]
        data = data[mask, :]


        # Plot rois
        for r in range(10):

            # Plot ROI
            plot(data_vars, states_vars, r)

            # Finalize plot
            title = f"{file.replace('_STATES.h5', '')} ROI={r}"
            fig = plt.gcf()
            fig.set_size_inches(8,5)
            fig.suptitle(title)
            plt.tight_layout()
            plt.pause(.1)

            # Save plot
            plt.savefig(f"pics/states/{title}.png")



    # Done
    print("Done!")
