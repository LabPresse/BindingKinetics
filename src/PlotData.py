
# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Define function
def plot_data(data, variables=None, groundtruth=None, merge=None):

    # Merge ROIs
    if merge is not None and merge > 1:
        """Merge ROIs together by summing the traces of <num_merged> ROIs together."""

        # Get the length of the new data after merging
        num_rois = num_rois // merge

        # Merge ROIs
        data_new = np.zeros((num_rois, num_frames))
        for i in range(num_rois):
            data_new[i, :] = np.sum(data[i*merge:(i+1)*merge, :], axis=0)

        # Update data
        data = data_new
        num_rois, num_frames = data.shape

    # Set up figure
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(data, "r", label="Data")

    # Plot variables
    if variables is not None:
        # Get variables
        s = variables.s
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        num_states = variables.num_states
        # Calculate trace
        trace = np.zeros_like(data)
        for k in range(num_states):
            ids = np.where(s == k)[0]
            trace[ids] = k * mu_flor + mu_back
        # Plot trace
        ax.plot(trace, "b", label=f"Trace\nMax={np.max(s)}")

    # Set up plot
    ax.set_ylabel("Intensity (ADU)")
    ax.set_xlabel("Time (Frame #)")
    ax.legend()
    plt.tight_layout()
    plt.pause(.1)

    # Finish
    return