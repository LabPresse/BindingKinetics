
# Import packages
import h5py
import warnings
import numpy as np

# Convert HDF5 to dictionary
def h5_to_dict(h5_obj):
    # Recursively convert an HDF5 file or group to a nested dictionary
    if isinstance(h5_obj, h5py.File):
        # If the object is a file, get its top-level attributes and convert them to a dictionary
        return {key: h5_to_dict(h5_obj[key]) for key in h5_obj.keys()}
    elif isinstance(h5_obj, h5py.Group):
        # If the object is a group, recursively convert all its members to a dictionary
        return {key: h5_to_dict(h5_obj[key]) for key in h5_obj.keys()}
    elif isinstance(h5_obj, h5py.Dataset):
        # If the object is a dataset, return its value
        return h5_to_dict(h5_obj[()])
    elif isinstance(h5_obj, bytes):
        # If the object is a byte array, convert it to a string
        return h5_obj.decode()
    elif isinstance(h5_obj, np.ndarray) and h5_obj.dtype=='O':
        # If the object is iteable, recursively convert all its elements
        for i, val in enumerate(h5_obj):
            h5_obj[i] = h5_to_dict(val)
        return h5_obj
    else:
        # If the object is anything else, return it as-is
        return h5_obj


    
# Convert dictionary to HDF5
def dict_to_h5(h5_obj, dict_obj):
    # Recursively convert a nested dictionary to an HDF5 file or group
    for key, val in dict_obj.items():
        if isinstance(val, dict):
            # If the value is a dictionary, recursively convert it to a group
            dict_to_h5(h5_obj.create_group(key), val)
        else:
            # If the value is not a dictionary, convert it to a dataset
            try:
                with warnings.catch_warnings():
                    h5_obj.create_dataset(key, data=val)
            except:
                h5_obj.create_dataset(key, data=f'<{type(val)}>'+val.__repr__())
    return


