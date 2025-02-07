
# Import libraries
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_rates(MAP, Samples, groundtruth=None):

    # Update MAP and groundtruth
    MAP = copy.deepcopy(MAP)
    if hasattr(MAP, "k_pb") and not hasattr(MAP, "k_photo"):
        MAP.k_photo = np.array([[-MAP.k_pb, MAP.k_pb], [0, 0]])
    if hasattr(MAP, "k_pb_shape") and not hasattr(MAP, "k_photo_shape"):
        MAP.k_photo_shape = MAP.k_pb_shape
    if hasattr(MAP, "k_pb_scale") and not hasattr(MAP, "k_photo_scale"):
        MAP.k_photo_scale = np.array([[-MAP.k_pb_scale, MAP.k_pb_scale], [0, 0]])
    groundtruth = copy.deepcopy(groundtruth)
    if hasattr(groundtruth, "k_pb") and not hasattr(groundtruth, "k_photo"):
        groundtruth.k_photo = np.array([[-groundtruth.k_pb, groundtruth.k_pb], [0, 0]])

    # Shift units
    MAP = copy.deepcopy(MAP)
    MAP.k_on = MAP.k_on * 1000
    MAP.k_on_scale = MAP.k_on_scale * 1000
    MAP.k_off = MAP.k_off * 1000
    MAP.k_off_scale = MAP.k_off_scale * 1000
    MAP.k_photo = MAP.k_photo * 1000
    MAP.k_photo_scale = MAP.k_photo_scale * 1000
    if groundtruth is not None:
        groundtruth.k_on = groundtruth.k_on * 1000
        groundtruth.k_off = groundtruth.k_off * 1000
        groundtruth.k_photo = groundtruth.k_photo * 1000
    Samples = copy.deepcopy(Samples)
    Samples["k_on"] = Samples["k_on"] * 1000
    Samples["k_off"] = Samples["k_off"] * 1000
    Samples["k_photo"] = Samples["k_photo"] * 1000

    # Set up figure
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    ax = np.empty(2+np.sum(MAP.k_photo > 0), dtype=object)
    for i in range(len(ax)):
        ax[i] = fig.add_subplot(1, len(ax), i+1)
        ax[i].set_yticks([])
        ax[i].set_xlabel("Rate (1/s)")
        ax[i].set_ylabel("Probability")
    ax[0].set_xlabel("Rate (1/s/nM)")
    ax[2].set_xlabel("Rate (1/s/W)")

    # Select sample ids
    num_iterations = len(Samples["P"])
    last = [*np.where(Samples["P"] == 0)[0], num_iterations][0]
    burn = int(last/2)
    
    # Set IDs
    bID = 0
    dID = 2
    pID = -1

    # Set colors
    colPRIOR = "y"
    colHIST = "b"
    colMAP = "g"
    colGT = "r"
    
    # Plot k on
    ks = Samples["k_on"][burn:last, bID]
    MAP_k = np.atleast_1d(MAP.k_on)[bID]
    xlim = [0, 1.5*np.max(ks)]
    title = f"K on = {MAP_k:.2e}"
    ax[0].hist(ks, color=colHIST, density=True, label="Samples")
    ax[0].axvline(MAP_k, color=colMAP, label="MAP")
    if groundtruth is not None and hasattr(groundtruth, "k_on"):
        GT_k = np.atleast_1d(groundtruth.k_on)[bID]
        title = title + f"\nGT = {GT_k:.2e}"
        ax[0].axvline(GT_k, color=colGT, label="Ground truth")
        xlim = [0, 1.5*np.max([np.max(ks), GT_k])]
    if hasattr(MAP, "k_on_shape") and hasattr(MAP, "k_on_scale") and (xlim[1] > xlim[0]):
        x = np.linspace(xlim[0], xlim[1], 1000)
        prior_shape = np.atleast_1d(MAP.k_on_shape)[bID]
        prior_scale = np.atleast_1d(MAP.k_on_scale)[bID]
        px = stats.gamma.pdf(x, a=prior_shape, scale=prior_scale)
        ax[0].plot(x, px, color=colPRIOR, label="Prior")
    ax[0].set_title(title)
    if xlim[1] > xlim[0]:
        ax[0].set_xlim(xlim)

    # Plot k off
    ks = Samples["k_off"][burn:last, bID]
    MAP_k = np.atleast_1d(MAP.k_off)[bID]
    xlim = [0, 1.5*np.max(ks)]
    title = f"K off = {MAP_k:.2e}"
    ax[1].hist(ks, color=colHIST, density=True, label="Samples")
    ax[1].axvline(MAP_k, color=colMAP, label="MAP")
    if groundtruth is not None and hasattr(groundtruth, "k_off"):
        GT_k = np.atleast_1d(groundtruth.k_off)[bID]
        title = title + f"\nGT = {GT_k:.2e}"
        ax[1].axvline(GT_k, color=colGT, label="Ground truth")
        xlim = [0, 1.5*np.max([np.max(ks), GT_k])]
    if hasattr(MAP, "k_off_shape") and hasattr(MAP, "k_off_scale") and (xlim[1] > xlim[0]):
        x = np.linspace(xlim[0], xlim[1], 1000)
        prior_shape = np.atleast_1d(MAP.k_off_shape)[bID]
        prior_scale = np.atleast_1d(MAP.k_off_scale)[bID]
        px = stats.gamma.pdf(x, a=prior_shape, scale=prior_scale)
        ax[1].plot(x, px, color=colPRIOR, label="Prior")
    ax[1].set_title(title)
    if xlim[1] > xlim[0]:
        ax[1].set_xlim(xlim)


    # Plot k bleach
    if "k_pb" in Samples.keys():
        ks = Samples["k_pb"][burn:last]
    elif "k_photo" in Samples.keys():
        ks = Samples["k_photo"][burn:last, bID, pID]
    MAP_k = np.atleast_2d(MAP.k_photo)[bID, pID]
    xlim = [0, 1.5*np.max(ks)]
    title = f"K bleach = {MAP_k:.2e}"
    ax[2].hist(ks, color=colHIST, density=True, label="Samples")
    ax[2].axvline(MAP_k, color=colMAP, label="MAP")
    if groundtruth is not None and hasattr(groundtruth, "k_photo"):
        GT_k = np.atleast_2d(groundtruth.k_photo)[bID, pID]
        title = title + f"\nGT = {GT_k:.2e}"
        ax[2].axvline(GT_k, color=colGT, label="Ground truth")
        xlim = [0, 1.5*np.max([np.max(ks), GT_k])]
    if hasattr(MAP, "k_photo_shape") and hasattr(MAP, "k_photo_scale") and (xlim[1] > xlim[0]):
        x = np.linspace(xlim[0], xlim[1], 1000)
        prior_shape = MAP.k_photo_shape[bID, pID]
        prior_scale = MAP.k_photo_scale[bID, pID]
        px = stats.gamma.pdf(x, a=prior_shape, scale=prior_scale)
        ax[2].plot(x, px, color=colPRIOR, label="Prior")
    ax[2].set_title(title)
    if xlim[1] > xlim[0]:
        ax[2].set_xlim(xlim)

    # Plot blink and unblink rates
    if MAP.k_photo.shape[0] == 4:

        # Blink
        ks = Samples["k_photo"][burn:last, bID, dID]
        MAP_k = MAP.k_photo[bID, dID]
        xlim = [0, 1.5*np.max(ks)]
        title = f"K blink = {MAP_k:.2e}"
        ax[3].hist(ks, color=colHIST, density=True, label="Samples")
        ax[3].axvline(MAP_k, color=colMAP, label="MAP")
        if groundtruth is not None and hasattr(groundtruth, "k_photo"):
            GT_k = groundtruth.k_photo[bID, dID]
            title = title + f"\nGT = {GT_k:.2e}"
            ax[3].axvline(GT_k, color=colGT, label="Ground truth")
            xlim = [0, 1.5*np.max([np.max(ks), GT_k])]
        if hasattr(MAP, "k_photo_shape") and hasattr(MAP, "k_photo_scale") and (xlim[1] > xlim[0]):
            x = np.linspace(xlim[0], xlim[1], 1000)
            prior_shape = MAP.k_photo_shape[bID, dID]
            prior_scale = MAP.k_photo_scale[bID, dID]
            px = stats.gamma.pdf(x, a=prior_shape, scale=prior_scale)
            ax[3].plot(x, px, color=colPRIOR, label="Prior")
        ax[3].set_title(title)
        if xlim[1] > xlim[0]:
            ax[3].set_xlim(xlim)
        
        # Unblink
        ks = Samples["k_photo"][burn:last, dID, bID]
        MAP_k = MAP.k_photo[dID, bID]
        xlim = [0, 1.5*np.max(ks)]
        title = f"K unblink = {MAP_k:.2e}"
        ax[4].hist(ks, color=colHIST, density=True, label="Samples")
        ax[4].axvline(MAP_k, color=colMAP, label="MAP")
        if groundtruth is not None and hasattr(groundtruth, "k_photo"):
            GT_k = groundtruth.k_photo[dID, bID]
            title = title + f"\nGT = {GT_k:.2e}"
            ax[4].axvline(GT_k, color=colGT, label="Ground truth")
            xlim = [0, 1.5*np.max([np.max(ks), GT_k])]
        if hasattr(MAP, "k_photo_shape") and hasattr(MAP, "k_photo_scale") and (xlim[1] > xlim[0]):
            x = np.linspace(xlim[0], xlim[1], 1000)
            prior_shape = MAP.k_photo_shape[dID, bID]
            prior_scale = MAP.k_photo_scale[dID, bID]
            px = stats.gamma.pdf(x, a=prior_shape, scale=prior_scale)
            ax[4].plot(x, px, color=colPRIOR, label="Prior")
        ax[4].set_title(title)
        if xlim[1] > xlim[0]:
            ax[4].set_xlim(xlim)

    # Finalize figure
    ax[-1].legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.pause(.01)
    return

