
# Import standard modules
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.LearnRates import RateInference
from src.dict2hdf5 import dict_to_h5, h5_to_dict

# Load configuration
from config import *

# Preprocess data
def preprocess(data, parameters, concentration=None, downsample=None, min_step=1, **kwargs):

    # Check if reversible
    print("Checking reversibility")
    if "experiment_parameters" in parameters.keys():
        if not parameters["experiment_parameters"]["reversible"]:
            parameters["k_on"] = 0
            parameters["k_off"] = 0
    if ("files" in parameters.keys()) and ("reversible" in parameters["files"][list(parameters["files"])[0]].keys()):
        if not parameters["files"][list(parameters["files"])[0]]["reversible"]:
            parameters["k_on"] = 0
            parameters["k_off"] = 0

    # Filter ROIs
    print("Filtering ROIs")
    mask = np.ones(data.shape[0], dtype=bool)
    for r in range(data.shape[0]):

        # Subtract background
        data[r, :] -= np.min(data[r, :])

        # Remove spikes that are shorter than min_step frames
        for s in range(np.max(data[r, :])+1):
            for dn in range(1, min_step+1):
                idx = np.where(
                    (data[r, :-(dn+1)] == s)
                    & (data[r, 1:-dn] != s)
                    & (data[r, dn+1:] == s)
                )[0]
                for i in idx:
                    data[r, i+1:i+dn+1] = s
        
        # Check if ROI has any states
        if np.all(data[r, :] == 0):
            mask[r] = False
            continue

        # Remove time levels where the state only lasts one frame
        if data[r, 0] != data[r, 1]:
            mask[r] = False
            continue
        for s in range(np.max(data[r, :])+1):
            idx = (
                (data[r, :-2] != s)
                & (data[r, 1:-1] == s)
                & (data[r, 2:] != s )
            )
            if np.any(idx):
                mask[r] = False
                break

        # Remove rois where the first state lasts at least min_step/2 frames
        if (data[r, :min_step//2] != data[r, 0]).any():
            mask[r] = False

    # Check concentration
    if concentration is not None:
        print(f"Filtering for concentration {concentration}")
        mask[parameters["concentration"] != concentration] = False
    
    if parameters["concentration"].shape[0] > mask.shape[0]:
        parameters["concentration"] = parameters["concentration"][:-1]
        parameters["laserpower"] = parameters["laserpower"][:-1]

    # Downsample
    if downsample is not None:
        print(f"Downsampling by {downsample}")
        newmask = np.zeros(data.shape[0], dtype=bool)
        SZtmp = min(mask.shape[0],parameters["laserpower"].shape[0])
        for c in np.unique(parameters["concentration"]):
            for l in np.unique(parameters["laserpower"]):
                ids = np.where(
                    mask[0:SZtmp]
                    & (parameters["concentration"][0:SZtmp] == c)
                    & (parameters["laserpower"][0:SZtmp]  == l)
                )[0]
                newmask[ids[::downsample]] = True
        mask = newmask

    # Apply mask
    print("Applying mask")
    data = data[mask]
    for key in parameters.keys():
        try:
            parameters[key] = parameters[key][mask]
        except:
            pass

    # Ensure correct key names
    if "concentrations" in parameters.keys():
        parameters["concentration"] = parameters["concentrations"]
        del parameters["concentrations"]
    if "laserpowers" in parameters.keys():
        parameters["laserpower"] = parameters["laserpowers"]
        del parameters["laserpowers"]

    # Store preprocessing parameters
    parameters["min_step"] = min_step
    parameters["statesmask"] = mask

    # Reset unnecsary parameters
    parameters['num_max'] = None

    # Return preprocessed data
    return data, parameters


# Main function
def main_rates(ID):

    # Set paths
    print(f"Setting paths - Current directory: {os.getcwd()}")
    datapath = os.path.join(PATH_DATA, "Binding/")
    savepath = os.path.join(PATH_DATA, "Binding/")

    # Create filesargs
    print("Selecting file and arguments")
    filesargs = [
        # ('simulated_Kon10-6_Koff25-6_2states_STATES.h5', {'min_step': 5, 'num_micro': 2}),
        # ('simulated_Kon20-6_Koff25-6_2states_STATES.h5', {'min_step': 5, 'num_micro': 2}),
        # ('simulated_Kon50-6_Koff25-6_2states_STATES.h5', {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=2e-6_koff=2e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=5e-6_koff=2e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=1e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=5e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=2e-5_kphoto=2e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=2e-5_kphoto=5e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2}),
        ("simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2, 'downsample': 2}),
        ("simulated_kon=1e-6_koff=2e-5_kphoto=1e-7_Nstates=2_STATES.h5", {'min_step': 5, 'num_micro': 2, 'downsample': 5}),
    ]
    file, args = filesargs[ID]

    # Load data
    print("Loading data")
    with h5py.File(datapath+file, "r") as h5:
        parameters = h5_to_dict(h5)
        data = h5["states"][()]
        del parameters["states"]

    # Preprocess data
    print("Preprocessing data")
    data, parameters = preprocess(data, parameters, **args)
    parameters["statesfile"] = file
    parameters["datamask"] = parameters["mask"]
    del parameters["mask"]
    
    # Analzye Rates
    print(f"Analyzing rates for {file}")
    savename = "_".join([
        file.replace("_STATES.h5", ""),
        "_".join([f"{key}={value}".replace("_", "") for key, value in sorted(args.items()) if value]),
        "RATES.h5",
    ])
    variables, Samples = RateInference.analyze(
        data,
        parameters,
        **args,
        plot=True,
        num_iterations=2000,
        savepath=savepath+savename,
        parallelize=False,
    )

    # Format output
    output = {
        **parameters,
        "variables": variables.__dict__,
        "samples": Samples,
        "datapath": datapath,
    }

    # Save results
    print("Saving results")
    with h5py.File(savepath+savename, "w") as h5:
        dict_to_h5(h5, output)

    # Done
    print(f"Finished with main_rates for {savename}")
    return


# Main script
if __name__ == "__main__":

    # Run
    for ID in range(9):
        main_rates(ID)

    print("Done")

