
# Import modules
import h5py
import copy
import numpy as np
from src.dict2hdf5 import h5_to_dict

def load_data(datafile, merge=None):

    # Load data and parameters
    with h5py.File(datafile, "r") as h5:
        parameters = h5_to_dict(h5)
        data = h5["data"][()]
        del parameters["data"]

    # Check for merge
    if merge is not None and merge > 1:

        # Initialize new data
        num_frames = data.shape[1]
        data_new = None
        mask_new = np.zeros(data.shape[0], dtype=bool)

        # Loop over settings
        laserpowers = parameters["laserpower"]
        concentrations = parameters["concentration"]
        settings = np.unique(np.array([concentrations, laserpowers]).T, axis=0)

        # Loop through settings
        num_rois = data.shape[0] // merge
        data_cl_new = np.zeros((num_rois, num_frames))
        NIntent = np.shape(settings)[0]
        ID = -1
        for c, l in settings:
            
            ID = ID + 1
            # Get indices
            idx = np.where((concentrations == c) & (laserpowers == l))[0]
            data_cl = data[idx, :]

            # Merge ROIs
            num_rois_cl = data_cl.shape[0] // merge
            for i in range(num_rois_cl):
                mask_new[idx[i*merge]] = True
                Ind = i*NIntent + ID
                if Ind < num_rois:
                    data_cl_new[Ind, :] = np.sum(data_cl[i*merge:(i+1)*merge, :], axis=0)

        # Append to data_new
        if data_new is None:
            data_new = data_cl_new
        else:
            data_new = np.vstack((data_new, data_cl))

        # Compress the ROIs in parameters
        parameters = copy.deepcopy(parameters)
        for key, val in parameters.items():
            try:
                parameters[key] = val[mask_new]
            except:
                pass
        data = data_new    
 
    # Return data and parameters
    return data, parameters

