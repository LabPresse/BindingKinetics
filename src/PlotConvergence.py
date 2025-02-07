
# Import libraries
import copy
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(MAP, Samples, groundtruth=None):

    # Update MAP and groundtruth
    MAP = copy.deepcopy(MAP)
    if hasattr(MAP, "k_pb") and not hasattr(MAP, "k_photo"):
        MAP.k_photo = np.array([[-MAP.k_pb, MAP.k_pb], [0, 0]])
    groundtruth = copy.deepcopy(groundtruth)
    if hasattr(groundtruth, "k_pb") and not hasattr(groundtruth, "k_photo"):
        groundtruth.k_photo = np.array([[-groundtruth.k_pb, groundtruth.k_pb], [0, 0]])

    # Set up figure
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    ax = np.empty(3+np.sum(MAP.k_photo > 0), dtype=object)
    for i in range(len(ax)):
        ax[i] = fig.add_subplot(1, len(ax), i+1)
        ax[i].set_yticks([])
        ax[i].set_xlabel("Iteration")
        ax[i].set_ylabel("Value")

    # Select sample ids
    num_iterations = len(Samples["P"])
    last = [*np.where(Samples["P"] == 0)[0], num_iterations][0]
    burn = int(last/2)
    iters = np.arange(burn, last)

    # Set colors
    colPLOT = "b"
    colGT = "r"
    
    # Plot k on
    ks = Samples["k_on"][burn:last]
    ax[0].set_title(f"K on")
    ax[0].plot(iters, ks, color=colPLOT, label="Samples")
    if groundtruth is not None and hasattr(groundtruth, "k_on"):
       ax[0].axhline(groundtruth.k_on[0], color=colGT, label="Ground truth")

    # Plot k off
    ks = Samples["k_off"][burn:last]
    ax[1].set_title(f"K off")
    ax[1].plot(iters, ks, color=colPLOT, label="Samples")
    if groundtruth is not None and hasattr(groundtruth, "k_off"):
        ax[1].axhline(groundtruth.k_off[0], color=colGT, label="Ground truth")

    # Plot k bleach
    if "k_pb" in Samples.keys():
        ks = Samples["k_pb"][burn:last]
    elif "k_photo" in Samples.keys():
        ks = Samples["k_photo"][burn:last, 0, -1]
    ax[2].set_title(f"K bleach")
    ax[2].plot(iters, ks, color=colPLOT, label="Samples")
    if groundtruth is not None and hasattr(groundtruth, "k_photo"):
        ax[2].axhline(groundtruth.k_photo[0, -1], color=colGT, label="Ground truth")

    # # Plot k blink and unblink
    # if MAP.k_photo.shape[0] == 3:

    #     # Blink
    #     ks = Samples["k_photo"][burn:last, 0, 1]
    #     ax[3].set_title(f"K blink")
    #     ax[3].plot(iters, ks, color=colPLOT, label="Samples")
    #     if groundtruth is not None and hasattr(groundtruth, "k_photo"):
    #         ax[3].axhline(groundtruth.k_photo[0, 0], color=colGT, label="Ground truth")
        
    #     # Unblink
    #     ks = Samples["k_photo"][burn:last, 0, 1]
    #     ax[4].set_title(f"K unblink")
    #     ax[4].plot(iters, ks, color=colPLOT, label="Samples")
    #     if groundtruth is not None and hasattr(groundtruth, "k_photo"):
    #         ax[4].axhline(groundtruth.k_photo[0, 1], color=colGT, label="Ground truth")

    # Plot log probability
    ax[-1].set_title(f"Log probability")
    ax[-1].set_ylabel("Log probability")
    ax[-1].plot(np.arange(len(Samples["P"])), Samples["P"], color=colPLOT, label="Samples")
    ax[-1].axvline(burn, color="k", linestyle="--", label="Burn")

    # Finalize figure
    ax[-1].legend()
    ax[-2].legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.pause(.01)
    return
