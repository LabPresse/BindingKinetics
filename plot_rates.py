
# Import libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from src.dict2hdf5 import h5_to_dict
from src.PlotData import plot_data
from src.PlotRates import plot_rates
from src.PlotConvergence import plot_convergence
from src.LearnRates import RateInference
from .cleandata import EXPERIMENT_PARAMETERS

# Load configuration
from config import PATH_DATA

# Define plot function
def plot(file, saveas=None, convergence=False, states=False):

    # Open file
    with h5py.File(file, "r") as h5:
        results = h5_to_dict(h5)
    MAP = SimpleNamespace(**results["variables"])
    Samples = results["samples"]
    if "groundtruth" in results:
        groundtruth = SimpleNamespace(**results["groundtruth"])
        if hasattr(groundtruth, "k_pb") and not hasattr(groundtruth, "k_photo"):
            groundtruth.k_photo = np.array([[-groundtruth.k_pb, groundtruth.k_pb], [0, 0]])
        elif hasattr(groundtruth, "k_photo") and not hasattr(groundtruth, "k_pb"):
            groundtruth.k_pb = groundtruth.k_photo[0, -1]
    else:
        groundtruth = None

    # Modify groundtruch is ROIs are merged
    if groundtruth is not None and 'merge=' in file:
        
        # Get num_merge from file name
        #                              file = ABC_merge=2_XYZ.h5   file = ABC_merge=2.h5
        a = file.split('merge=')[1]  #    a = 2_XYZ.h5                a = 2.h5
        b = a.split('_')[0]          #    b = 2                       b = 2.h5
        c = b.split('.')[0]          #    c = 2                       c = 2
        num_merge = int(c)

        # Update groundtruth according to num_merge
        groundtruth.k_on *= num_merge
    
    # Plot
    if convergence:
        plot_convergence(MAP, Samples, groundtruth=groundtruth)
    elif states:
        datapath = os.path.join(PATH_DATA, 'Binding', results["statesfile"])
        with h5py.File(datapath, "r") as h5:
            data = h5["states"][()]
        data = data[results['statesmask'], :]
        MAP.lhood_dict = tuple([
            np.where(np.sum(MAP.partitions[:, MAP.bright_ids], axis=1) == i)[0] for i in range(MAP.num_max+1)
        ])
        RateInference.plot_states(data, MAP, r=0)
    else:
        plot_rates(MAP, Samples, groundtruth=groundtruth)

    # Set title
    title = file.split("/")[-1]

    # Finalize plot
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    fig.suptitle(title)
    plt.tight_layout()
    plt.pause(.1)

    # Save plot
    if saveas is not None:
        plt.savefig(saveas, dpi=300)
    
    # Return
    return


# Main script
if __name__ == "__main__":

    # Get files
    path = "outfiles/"
    files = os.listdir(path)
    files = [f for f in files if f.endswith("RATES.h5")]

    # Loop through files
    for file in files:
        plot(path+file, f"pics/rates/{file[:-3]}.png")
        plot(path+file, f"pics/states/{file[:-3]}.png", states=True)
        plot(path+file, f"pics/convergence/{file[:-3]}.png", convergence=True)
        plt.pause(.1)

    # Done
    print("Done")
