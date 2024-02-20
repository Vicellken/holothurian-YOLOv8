import matplotlib.pyplot as plt
import pandas as pd

# load the data
box = pd.read_csv("loss_nano.csv", usecols=[0, 1])
dfl = pd.read_csv("loss_nano.csv", usecols=[0, 2])
mAP = pd.read_csv("mAP_nano.csv", usecols=[0, 1])
mAP95 = pd.read_csv("mAP_nano.csv", usecols=[0, 2])

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharex=True,
    figsize=(14, 6),
    # gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

# First plot (loss)
ax1.plot("epoch", "n_dfl", data=dfl, label="DFL (distribution focal loss)")
ax1.plot(
    "epoch",
    "n_box",
    data=box,
    label="$L_{CIoU}$ (complete intersection over union loss)",
)
ax1.set_ylabel("Loss", fontsize=14, fontweight="bold")
ax1.legend(loc="best", fontsize=14)

# Add label 'a' to the first plot
ax1.text(-0.15, 1.05, "a", transform=ax1.transAxes,
         fontsize=16, fontweight="bold")
ax1.tick_params(axis='both', which='major', labelsize=16)

# Second plot (mAP)
ax2.plot("epoch", "n_mAP", data=mAP, label="mAP")
ax2.plot("epoch", "n_mAP95", data=mAP95, label="mAP50-95")
ax2.set_ylabel("mAP", fontsize=14, fontweight="bold")
ax2.legend(loc="best", fontsize=14)

# Add label 'b' to the second plot
ax2.text(-0.15, 1.05, "b", transform=ax2.transAxes,
         fontsize=16, fontweight="bold")
ax2.tick_params(axis='both', which='major', labelsize=16)

# Set the layout
# fig.suptitle("Metrics for the nano dataset", fontsize=14, fontweight="bold")
fig.supxlabel("Number of train iterations", fontsize=14, fontweight="bold")

# Save the figure
plt.savefig(
    "metrics_nano.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
