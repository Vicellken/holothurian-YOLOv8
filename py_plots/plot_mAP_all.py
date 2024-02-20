import matplotlib.pyplot as plt
import pandas as pd

# load the data
mAP_nano = pd.read_csv("mAP_all_csv.csv", usecols=[0, 1])
mAP95_nano = pd.read_csv("mAP_all_csv.csv", usecols=[0, 2])
mAP_s = pd.read_csv("mAP_all_csv.csv", usecols=[0, 3])
mAP95_s = pd.read_csv("mAP_all_csv.csv", usecols=[0, 4])
mAP_m = pd.read_csv("mAP_all_csv.csv", usecols=[0, 5])
mAP95_m = pd.read_csv("mAP_all_csv.csv", usecols=[0, 6])
mAP_l = pd.read_csv("mAP_all_csv.csv", usecols=[0, 7])
mAP95_l = pd.read_csv("mAP_all_csv.csv", usecols=[0, 8])
mAP_xl = pd.read_csv("mAP_all_csv.csv", usecols=[0, 9])
mAP95_xl = pd.read_csv("mAP_all_csv.csv", usecols=[0, 10])

# Plot the loss and mAP for the nano dataset
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharex=True,
    figsize=(15, 7),
    constrained_layout=True,
)

# First plot (mAP)
ax1.plot("epoch", "n_mAP", data=mAP_nano, label="Nano mAP")
ax1.plot("epoch", "s_mAP", data=mAP_s, label="Small mAP")
ax1.plot("epoch", "m_mAP", data=mAP_m, label="Medium mAP")
ax1.plot("epoch", "l_mAP", data=mAP_l, label="Large mAP")
ax1.plot("epoch", "x_mAP", data=mAP_xl, label="XL mAP")
ax1.set_ylabel("mAP", fontsize=14, fontweight="bold")
ax1.legend(loc="best", fontsize=14)

# Add label 'a' to the first plot
ax1.text(
    0.1,
    1.05,
    "a. mean average precision",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
)
ax1.tick_params(axis="both", which="major", labelsize=16)

# Second plot (mAP95)
ax2.plot("epoch", "n_mAP95", data=mAP95_nano, label="Nano mAP50-95")
ax2.plot("epoch", "s_mAP95", data=mAP95_s, label="Small mAP50-95")
ax2.plot("epoch", "m_mAP95", data=mAP95_m, label="Medium mAP50-95")
ax2.plot("epoch", "l_mAP95", data=mAP95_l, label="Large mAP50-95")
ax2.plot("epoch", "x_mAP95", data=mAP95_xl, label="XL mAP50-95")
ax2.set_ylabel("mAP", fontsize=14, fontweight="bold")
ax2.legend(loc="best", fontsize=14)

# Add label 'b' to the second plot
ax2.text(
    0.1,
    1.05,
    "b. mean average precision 50-95",
    transform=ax2.transAxes,
    fontsize=16,
    fontweight="bold",
)
ax2.tick_params(axis="both", which="major", labelsize=16)

# Set the layout
fig.supxlabel("Number of train iterations", fontsize=14, fontweight="bold")

# Save the figure
plt.savefig(
    "mAP_all.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
