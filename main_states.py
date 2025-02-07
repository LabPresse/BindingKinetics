
# Import standard modules
import os
import sys
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.LearnStates import StateInference
from src.dict2hdf5 import dict_to_h5, h5_to_dict
from src.load_data import load_data

# Load configuration
from config import *

# Find states
def find_states(data, parameters, parallelize=True, verbose=False, plot=False, **kwargs):

    # Filter out bad ROIs
    mask = (
        (np.isnan(data).sum(axis=1) == 0)  # No NaNs
        &
        ((data == 0).sum(axis=1) == 0)  # No zeros
    )
    data = data[mask, :]
    for key in parameters.keys():
        try:
            parameters[key] = parameters[key][mask]
        except:
            pass
    num_rois, num_frames = data.shape

    # Analyze states
    print("Analyzing states")

    # Create function to analyze
    def analyze(r):
        
        # Initialize constants
        t = time.time()
        n_tries = 0

        # Analyze ROI in try-except loop
        while n_tries < 10:
            try:
                variables = StateInference.analyze(data[r, :], parameters, verbose=verbose, plot=plot, **kwargs)
                break
            except:
                n_tries += 1
                print(f"Parallelization failed for ROI {r}. Trying again ({n_tries}/10)")
        
        # Format output
        output = {
            "states": variables.s,
            "mu_flor": variables.mu_flor,
            "mu_back": variables.mu_back,
            "sigma_flor": variables.sigma_flor,
            "sigma_back": variables.sigma_back,
        }

        # Print progress
        if r % 10 == 0:
            print(f"- {r}/{num_rois} ({time.time()-t:.1f} s per ROI)", flush=True)

        # Return output
        return output
    
    # Run analysis
    if parallelize:
        results = Parallel(n_jobs=10)(delayed(analyze)(r) for r in range(num_rois))
    else:
        results = [analyze(r) for r in range(num_rois)]

    # Organize results
    output = {
        **parameters,
        "parameters": parameters,
        "mask": np.where(mask)[0],
        "states": np.array([x["states"] for x in results], dtype=int),
        "mu_flor": np.array([x["mu_flor"] for x in results]),
        "mu_back": np.array([x["mu_back"] for x in results]),
        "sigma_flor": np.array([x["sigma_flor"] for x in results]),
        "sigma_back": np.array([x["sigma_back"] for x in results]),
    }

    # Return output
    return output

# Main funtion
def main_states(ID=0):

    # Set paths
    print(f"Setting paths - Current directory: {os.getcwd()}")
    datapath = os.path.join(PATH_DATA, "/Binding/")
    savepath = os.path.join(PATH_DATA, "/Binding/")

    # Get files
    print("Selecting file")
    files = [
        # 'simulated_Kon10-6_Koff25-6_2states.h5',
        # 'simulated_Kon20-6_Koff25-6_2states.h5',
        # 'simulated_Kon50-6_Koff25-6_2states.h5',
        "simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2.h5",
        "simulated_kon=2e-6_koff=2e-5_kphoto=1e-7_Nstates=2.h5",
        "simulated_kon=5e-6_koff=2e-5_kphoto=1e-7_Nstates=2.h5",
        "simulated_kon=1e-6_koff=1e-5_kphoto=1e-7_Nstates=2.h5",
        "simulated_kon=1e-6_koff=5e-5_kphoto=1e-7_Nstates=2.h5",
        "simulated_kon=1e-6_koff=2e-5_kphoto=2e-7_Nstates=2.h5",
        "simulated_kon=1e-6_koff=2e-5_kphoto=5e-7_Nstates=2.h5",
    ]

    # Get arguments
    print("Getting arguments")
    runargs = [{}]

    # Select file and arguments
    print("Selecting file and arguments")
    filesargs = [(file, args) for file in files for args in runargs]
    file, args = filesargs[ID]

    # Load data
    print("Loading data")
    merge = args.get("merge", None)
    data, parameters = load_data(datapath+file, merge=merge)
    data -= np.min(data)

    # Analyze States
    output = find_states(
        data,
        parameters,
        plot=True,
        # verbose=True,
        # parallelize=False,
        num_iterations=100,
        **args,
    )
    output["datafile"] = file

    # Save results
    print("Saving results")
    savebase = file.replace(".h5", "")
    saveargs = "_".join([f"{key}={value}".replace('_', '') for key, value in sorted(args.items()) if value])
    if len(saveargs) > 0:
        savebase = savebase + "_" + saveargs
    with h5py.File(savepath+savebase+"_STATES.h5", "w") as h5:
        dict_to_h5(h5, output)

    # Done
    print(f"Done with {savebase}")
    return

# Main script
if __name__ == "__main__":

    # Run
    for ID in range(7):
        main_states(ID)

    # Done
    print("Done")

