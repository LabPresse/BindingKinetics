
# Import libraries
import os
import h5py
import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dict2hdf5 import dict_to_h5, h5_to_dict

# Load configuration
from config import PATH_DATA

# Set up path
path = os.path.join(PATH_DATA, "Binding/Raw/Results_corrected_new")

EXPERIMENT_PARAMETERS = {
    # Shared_Presse_Herten\ST089_230114_Buffers_on_photophysics and Shared_Presse_Herten\ST089_Buffers_on_photophysics- Bleaching and blinking kinetics recorded at different framerates.
    # Imager strand length:21 nucleotide - permanent binding
    # Relevan folder is :Shared_Presse_Herten\ST089_Buffers_on_photophysics\results_max
    # Inner Folder style:
    # LT1A1_B1_B2_A2_XXXpct_photophysics/bleaching
    # XXX-laser power
    # photophysics-recorded at 20ms exposure with frame transfer (faster mode where the lines are exposed individually not as a whole frame)
    # bleaching- recorded at 200ms exposure with no frame transfer
    # Subfolder style:
    # LTRC_BASE_OS
    # LT-labtek
    # R-Row (A-B)
    # C-Column (1-4)
    # BASE-ROXS (contains Mg) or Michi (contains higher NaCl and tween20)
    # ROXS- 5xPBS 12.5% glycerol 20mM MgCl2
    # OR
    # Michi- 700 NaCl , 0.05% TritonX100  <---We use this one for the Dynamic Measurements
    # AND
    # GOC-Glucose Oxidase+Catalase
    # OR
    # PCD- Protochatecuic Acid+Protocatecuate Dehydrogenase <---We use this one for previous static Measurements
    # !!!!NB Some of the results have an issue where the files created an issue with how they were saved 
    # !!!!NB versus how they were analysed. This was the issue with having imaging sets above 4GB which are
    # !!!!NB then separated into 4GB file of about 8000 frames. This has led to the extraction function to 
    # !!!!NB fill out the files with zeroes to the whatever is said to be the maximum for some data that
    # !!!!NB would be 20000 or 40000 frame for some data sets. You might need to specify that you want the 
    # !!!!NB data to only frame 8000.
    # 21 nucleotides
    # 1 binding site
    "ST089": {
        "reversible": False,
        "nucs": 21,
        "buffer": "Multiple",
        "sites": 1,
        "dt": [20, 200],
        "laserpower": [10, 50, 100],
        "comments": "Different buffers to examine potential changes in bleaching/blinking",
    },
   
    # Shared_Presse_Herten\ST090_230123_Dynamic_DNA_origami-Bleaching in statically bound DNA origami in different concentrations of imager.
    # Imager strand length:21 nucleotide - permanent binding
    # Relevant folder is :Shared_Presse_Herten\ST090_230123_Dynamic_DNA_origami\photobleaching\ST090_230123_Statically_bound_in_solution_DNA_origami
    # Inner Folder style:
    # LT1A2_A3_B2_B3_bleaching_with_dye_xxxpct
    # xxx-Laser power
    # All files are recorded with 100ms exposure (124ms frame time).
    # 21 nucleotides
    # 1 binding sites
   "ST090": {
        "reversible": False,
        "nucs": 21,
        "buffer": "ROXS Red (Michi)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": None,
        "comments": "Static experiment in different concentrations of imager",
    },

    # Shared_Presse_Herten\ST092_230124_Dynamic_DNA_origami_experiments_1BS-One binding site in optimal conditions
    # Imager strand length:9 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST092_230124_Dynamic_DNA_origami_experiments_1BS\photobleaching\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect little blinking. Mostly binding and unbinding and bleaching.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 9 nucleotides
    # 1 binding sites
    "ST092": {
        "reversible": True,
        "nucs": 9,
        "buffer": "ROXS Red (Michi)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "One binding site bindning and unbinding in optimal conditions",
    },

    # Shared_Presse_Herten\ST097_Dynamic_DNA_No_quenchers-One binding site experiment with no triplet quenchers.
    # Imager strand length:9 nucleotide - reversible binding
    # Expected blinking and slightly quicker bleaching.
    # Relevant folder is: Shared_Presse_Herten\ST097_Dynamic_DNA_No_quenchers\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect Lots of blinking fast bleaching
    # Michi-PCD - with no triplet quenchers
    # 9 nucleotides
    # 1 binding sites
    "ST097": {
        "reversible": True,
        "nucs": 9,
        "buffer": "ROXS Red (Michi) w/o MV AA",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "One binding site bindning and unbinding w/o quenchers expect blinking",
    },

    # Shared_Presse_Herten\ST098_Dynamic_DNA_base_only\One binding site experiment in PBS.
    # Imager strand length:9 nucleotide - reversible binding
    # Expected really quick bleaching.
    # Relevant folder is: Shared_Presse_Herten\ST098_Dynamic_DNA_base_only\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect fast bleaching maybe less blinking
    # PBS buffer
    # 9 nucleotides
    # 1 binding sites
    "ST098": {
        "reversible": True,
        "nucs": 9,
        "buffer": "PBS (extra NaCl)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "One binding site bindning and unbinding in PBS expect fast bleaching",
    },
    
    # Shared_Presse_Herten\ST100_230217_Dynamic_DNA_5bs-Five binding sites in optimal conditions
    # Relevant folder is: Shared_Presse_Herten\ST100_230217_Dynamic_DNA_5bs\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect similar blinking rates to ST089-Michi-PCD
    # Michi-PCD buffer
    # 9 nucleotides
    # 5 binding sites
    "ST100": {
        "reversible": True,
        "nucs": 9,
        "buffer": "ROXS Red (Michi)",
        "sites": 5,
        "dt": 124,
        "laserpower": [50, 500],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "Five binding site bindning and unbinding in optimal conditions",
    },
    
    # Shared_Presse_Herten\ST101_230224_Dynamic_binding_2BS-Two binding sites in optimal conditions
    # Imager strand length:9 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST101_230224_Dynamic_binding_2BS\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect similar blinking rates to ST089-Michi-PCD
    # Michi-PCD buffer
    # 9 nucleotides
    # 2 binding sites
    "ST101": {
        "reversible": True,
        "nucs": 9,
        "buffer": "ROXS Red (Michi)",
        "sites": 2,
        "dt": 124,
        "laserpower": [50, 200],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "Two binding site binding and unbinding in optimal conditions",
    },
    
    # Shared_Presse_Herten\ST113_140423_Dynamic_binding_1bs_repeat-One binding site in optimal conditions reversed imaging
    # NB!-Concentrations recorded in reveresed order 10nM,5nM,2nM,1nM. Repeat of ST092 in all other respect.
    # Imager strand length:9 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST113_140423_Dynamic_binding_1bs_repeat\LT1B3B2A3A2\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=10nM,5-8=5nM,9-12=2nM,13-16=1nM.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 9 nucleotides
    # 1 binding sites
    "ST113": {
        "reversible": True,
        "nucs": 9,
        "buffer": "ROXS Red (Michi)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=10nM,5-8=5nM,9-12=2nM,13-16=1nM",
        "comments": "One binding site bindning and unbinding in optimal conditions. Order of imaging is reversed to explore effects of time on buffer",
    },
    
    # Shared_Presse_Herten\ST114_240423_Dynamic_binding_1bs_10nt-One binding site in optimal conditions
    # Imager strand length:10 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST113_140423_Dynamic_binding_1bs_repeat\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface6-9=1nM,10-13=2nM,14-17=5nM,18-21=10nM.
    # Expect Unbinding expected to be slower.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 10 nucleotides
    # 1 binding sites
    "ST114": {
        "reversible": True,
        "nucs": 10,
        "buffer": "ROXS Red (Michi)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface6-9=1nM,10-13=2nM,14-17=5nM,18-21=10nM",
        "comments": "One binding site bindning and unbinding in optimal conditions with slower unbinding. Images start from surface 06. can be changed if needed",
    },

    # Shared_Presse_Herten\ST115_250423_Dynamic_binding_1bs_8nt-One binding site in optimal conditions
    # Imager strand length:10 nucleotide - reversible binding
    # NB!-Do not use surface 3 for analysis as it is out of focus.
    # Relevant folder is: Shared_Presse_Herten\ST115_250423_Dynamic_binding_1bs_8nt\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect Unbinding expected to be faster.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 8 nucleotides
    # 1 binding sites
    "ST115": {
        "reversible": True,
        "nucs": 8,
        "buffer": "ROXS Red (Michi)",
        "sites": 1,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "One binding site bindning and unbinding in optimal conditions with faster unbinding.",
    },

    # Shared_Presse_Herten\ST128_240623_Dynamic_DNA_2BS_10nt-Two binding site in optimal conditions
    # Imager strand length:10 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST128_240623_Dynamic_DNA_2BS_10nt\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect Unbinding expected to be slower.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 10 nucleotides
    # 2 binding sites
    "ST128": {
        "reversible": True,
        "nucs": 10,
        "buffer": "ROXS Red (Michi)",
        "sites": 2,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "Two binding site bindning and unbinding in optimal conditions",
    },

    # Shared_Presse_Herten\ST129_250623_Dynamic_DNA_5BS_10nt-Five binding site in optimal conditions
    # Imager strand length:10 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST129_250623_Dynamic_DNA_5BS_10nt\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface6-9=1nM,10-13=2nM,14-17=5nM,18-21=10nM.
    # Expect Unbinding expected to be slower.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 10 nucleotides
    # 5 binding sites
    "ST129": {
        "reversible": True,
        "nucs": 10,
        "buffer": "ROXS Red (Michi)",
        "sites": 5,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface6-9=1nM,10-13=2nM,14-17=5nM,18-21=10nM",
        "comments": "Five binding site bindning and unbinding in optimal conditions",
    },

    # Shared_Presse_Herten\ST130_250623_Dynamic_DNA_2BS_8nt-Two binding site in optimal conditions
    # Imager strand length:8 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST130_250623_Dynamic_DNA_2BS_8nt\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect Unbinding expected to be faster.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 8 nucleotides
    # 2 binding sites
    # NB-Surface 9 and 13 were not included as autofocus was not on
    "ST130": {
        "reversible": True,
        "nucs": 8,
        "buffer": "ROXS Red (Michi)",
        "sites": 2,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "Two binding site bindning and unbinding in optimal conditions",
    },

    # Shared_Presse_Herten\ST131_250623_Dynamic_DNA_5BS_8nt-Five binding site in optimal conditions
    # Imager strand length:8 nucleotide - reversible binding
    # Relevant folder is: Shared_Presse_Herten\ST131_250623_Dynamic_DNA_5BS_8nt\LT1A2A3B2B3\results_max
    # Files inside are different concentrations grouped in 4- surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM.
    # Expect Unbinding expected to be faster.
    # Should have same blinking and bleaching rates as ST089-Michi-PCD
    # Michi-PCD
    # 8 nucleotides
    # 5 binding sites
    # NB-Surface 9 and 13 were not included as autofocus was not on
    "ST131": {
        "reversible": True,
        "nucs": 8,
        "buffer": "ROXS Red (Michi)",
        "sites": 5,
        "dt": 124,
        "laserpower": [50, 100],
        "concentration_index": "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM",
        "comments": "Five binding site bindning and unbinding in optimal conditions",
    },

}


# Convert concentration index into list of concentrations
concentration_index = "surface1-4=1nM,5-8=2nM,9-12=5nM,13-16=10nM"
concentration_index = concentration_index[7:].split(',')
concentration_index = {x.split('=')[0]: x.split('=')[1] for x in concentration_index}
concentration_index = {key: int(value[:-2]) for key, value in concentration_index.items()}
concentration_index = {
    x: val
    for key, val in concentration_index.items()
    for x in range(int(key.split('-')[0]), int(key.split('-')[1])+1)
}


# Create file parser
def load_file(file):

    # Parse file name
    """
    General trace file function:
    SurfaceXX_YYY_PPP_WW_GG_peak/difference/background.csv
    XX = surface number (see below)
    YYY = laser wavelength 
    PPP = power in percentage of max power 10/50/100%
    WW = exposure  mostly 100ms - 124ms frame time
    GG = EM gain - 20
    """
    layout = "surface{surface}_{wavelength}_{laserpower}_{dt}_{gain}_difference_corrected.csv"
    parameters = parse.parse(layout, file.split('/')[-1]).named

    # Load dataframe
    try:
        df = pd.read_csv(file)
    except:
        print(f"Error loading {file}")
        return None, parameters

    # Extract data
    data = df.iloc[:, 15:8005].values
    
    # Filter based on reference distance
    dx = df['reference_distance']
    data = data[dx < 200]

    # Return data and parameters
    return data, parameters

# Merge experiment
def merge_experiment(experiment):

    # Set up data and parameters
    data = []
    exp_parameters = EXPERIMENT_PARAMETERS[experiment.split('_')[0]]
    parameters = {
        **exp_parameters,
        'concentration': [],
        'laserpower': [],
    }

    # Get concentration index
    concentration_index = exp_parameters['concentration_index']
    concentration_index = concentration_index[7:].split(',')
    concentration_index = {x.split('=')[0]: x.split('=')[1] for x in concentration_index}
    concentration_index = {key: int(value[:-2]) for key, value in concentration_index.items()}
    concentration_index = {
        x: val
        for key, val in concentration_index.items()
        for x in range(int(key.split('-')[0]), int(key.split('-')[1])+1)
    }

    # Loop through files
    for file in os.listdir(f"{path}/{experiment}"):
        if not file.endswith("corrected.csv"):
            continue
        
        # Load file
        filedata, fileparameters = load_file(f"{path}/{experiment}/{file}")
        if filedata is None:
            continue

        # Add laser power and concentration
        l = np.ones(filedata.shape[0]) * float(fileparameters['laserpower'])
        c = np.ones(filedata.shape[0]) * concentration_index[int(fileparameters["surface"])]

        # Add data
        data.append(filedata)
        parameters['laserpower'].append(l)
        parameters['concentration'].append(c)

    # Ensure data is same shape
    num_frames = min(d.shape[1] for d in data)
    data = [d[:, :num_frames] for d in data]

    # Merge data
    data = np.concatenate(data, axis=0)
    parameters['concentration'] = np.concatenate(parameters['concentration'], axis=0)
    parameters['laserpower'] = np.concatenate(parameters['laserpower'], axis=0)

    # Return data and parameters
    return data, parameters



# Main
if __name__ == "__main__":

    # Clean Experiments
    experiments = [
        # 'ST092_230124_Dynamic_DNA_origami_experiments_1BS',
        # 'ST100_230217_Dynamic_DNA_5bs',
        # 'ST101_230224_Dynamic_binding_2BS',
        # 'ST113_140423_Dynamic_binding_1bs_repeat',
        'ST114_240423_Dynamic_binding_1bs_10nt',
        # 'ST115_250423_Dynamic_binding_1bs_8nt',
        'ST128_240623_Dynamic_DNA_2BS_10nt',
        'ST129_250623_Dynamic_DNA_5BS_10nt',
        # 'ST130_250623_Dynamic_DNA_2BS_8nt', 
        # 'ST131_250623_Dynamic_DNA_5BS_8nt',
    ]
    for exp in experiments:
        print(f"Cleaning {exp}...")

        # Get data and parameters
        data, parameters = merge_experiment(exp)
        output = {
            'data': data,
            **parameters,
        }

        # Save data
        savepath = f'{os.environ["DATAPATH"]}/Binding/{exp.split("_")[0]}_filtered.h5'
        with h5py.File(savepath, "w") as h5:
            dict_to_h5(h5, output)

    # Done
    print("Done!")


