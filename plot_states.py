
# Import libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from src.dict2hdf5 import h5_to_dict
from src.load_data import load_data

# Load configuration
from config import PATH_DATA

# Define plot function
def plot(datafile, resultsfile=None, rois=None, merge=None, saveas=None):

    # Load data
    merge = 1
    data, parameters = load_data(datafile, merge=merge)

    # Load results
    if resultsfile is not None:
        with h5py.File(resultsfile, "r") as h5:
            results = h5_to_dict(h5)
        num_rois, num_frames = results["states"].shape
    else:
        results = None
    
    # Set rois
    rois = np.array([0, 1, 2, 3, 4, 5])
    if rois is None:
        # rois = np.arange(0, num_rois-1, num_rois//5)
        rois = np.arange(np.min((5, data.shape[0])))

    # Set up figure
    plt.close()
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    ax = np.empty(len(rois), dtype=object)
    for i, r in enumerate(rois):
        ax[i] = fig.add_subplot(len(rois), 1, i+1)
    
    # Plot
    for i, r in enumerate(rois):

        # Plot data
        ax[i].plot(data[r], "r", label="Data")

        # Plot variables
        if results is not None:
            # Get roi
            q = np.where(results["mask"])[0][r]
            # Get variables
            s = results["states"][q]
            mu_flor = results["mu_flor"][q]
            mu_back = results["mu_back"][q]
            # Calculate trace
            trace = np.zeros(data.shape[1])
            for k in range(np.max(s)+1):
                ids = np.where(s == k)[0]
                trace[ids] = k * mu_flor + mu_back
            # Plot trace
            ax[i].plot(trace, "b", label=f"Trace\nMax={np.max(s)}")

    # Finalize plot
    fig.suptitle(datafile.split("/")[-1].split(".")[0])
    fig.set_size_inches(10, 6)
    ax[-1].set_xlabel("Time (frames)")
    for i in range(len(rois)):
        ax[i].legend()
        ax[i].set_ylabel("Intensity (AU)")
    plt.tight_layout()
    plt.pause(.1)

    # Save plot
    if saveas is not None:
        plt.savefig(saveas, dpi=300)

    # Return
    return


# Main script
if __name__ == "__main__":

    # Set paths
    datapath = os.path.join(PATH_DATA, "Binding/")

    # Get data files
    datafiles = [f for f in os.listdir(datapath) if f.endswith(".h5")]
    datafiles = [f for f in datafiles if not f.endswith("STATES.h5")]
    #datafiles = [f for f in datafiles if f.startswith("ST089")]

    # Get results files
    resultsfiles = [f for f in os.listdir(datapath) if f.endswith("STATES.h5")]
    #resultsfiles = [f for f in resultsfiles if "num_states=2" in f]

    # Plot
    for f in datafiles:
        for x in resultsfiles:
            if f.split(".")[0] in x:
                plot(
                    datapath+f,
                    datapath+x,
                    saveas=f"pics/{x[:-3]}.png"
                ) 
                plt.pause(0.1)

    # Done
    print("Done!")



