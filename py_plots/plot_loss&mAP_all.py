import matplotlib.pyplot as plt
import pandas as pd

# load the data
box_nano = pd.read_csv("loss_all_csv.csv", usecols=[0, 1])
dfl_nano = pd.read_csv("loss_all_csv.csv", usecols=[0, 2])
mAP_nano = pd.read_csv("mAP_all_csv.csv", usecols=[0, 1])
mAP95_nano = pd.read_csv("mAP_all_csv.csv", usecols=[0, 2])
box_s = pd.read_csv("loss_all_csv.csv", usecols=[0, 3])
dfl_s = pd.read_csv("loss_all_csv.csv", usecols=[0, 4])
mAP_s = pd.read_csv("mAP_all_csv.csv", usecols=[0, 3])
mAP95_s = pd.read_csv("mAP_all_csv.csv", usecols=[0, 4])
box_m = pd.read_csv("loss_all_csv.csv", usecols=[0, 5])
dfl_m = pd.read_csv("loss_all_csv.csv", usecols=[0, 6])
mAP_m = pd.read_csv("mAP_all_csv.csv", usecols=[0, 5])
mAP95_m = pd.read_csv("mAP_all_csv.csv", usecols=[0, 6])
box_l = pd.read_csv("loss_all_csv.csv", usecols=[0, 7])
dfl_l = pd.read_csv("loss_all_csv.csv", usecols=[0, 8])
mAP_l = pd.read_csv("mAP_all_csv.csv", usecols=[0, 7])
mAP95_l = pd.read_csv("mAP_all_csv.csv", usecols=[0, 8])
box_xl = pd.read_csv("loss_all_csv.csv", usecols=[0, 9])
dfl_xl = pd.read_csv("loss_all_csv.csv", usecols=[0, 10])
mAP_xl = pd.read_csv("mAP_all_csv.csv", usecols=[0, 9])
mAP95_xl = pd.read_csv("mAP_all_csv.csv", usecols=[0, 10])

# Plot the loss and mAP for the nano dataset
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharex=True,
    figsize=(15, 7),
    # gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

# First plot (loss)
ax1.plot("epoch", "n_dfl", data=dfl_nano, label="Nano DFL")
ax1.plot("epoch", "n_box", data=box_nano, label="Nano $L_{CIoU}$")
ax1.plot("epoch", "s_dfl", data=dfl_s, label="Small DFL")
ax1.plot("epoch", "s_box", data=box_s, label="Small $L_{CIoU}$")
ax1.plot("epoch", "m_dfl", data=dfl_m, label="Medium DFL")
ax1.plot("epoch", "m_box", data=box_m, label="Medium $L_{CIoU}$")
ax1.plot("epoch", "l_dfl", data=dfl_l, label="Large DFL")
ax1.plot("epoch", "l_box", data=box_l, label="Large $L_{CIoU}$")
ax1.plot("epoch", "x_dfl", data=dfl_xl, label="XL DFL")
ax1.plot("epoch", "x_box", data=box_xl, label="XL $L_{CIoU}$")
ax1.set_ylabel("Loss", fontsize=14, fontweight="bold")
ax1.legend(loc="best", fontsize=14)

# Add label 'a' to the first plot
ax1.text(
    -0.15,
    1.05,
    "a",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
)
ax1.tick_params(axis="both", which="major", labelsize=16)

# Second plot (mAP)
ax2.plot("epoch", "n_mAP", data=mAP_nano, label="Nano mAP")
ax2.plot("epoch", "n_mAP95", data=mAP95_nano, label="Nano mAP50-95")
ax2.plot("epoch", "s_mAP", data=mAP_s, label="Small mAP")
ax2.plot("epoch", "s_mAP95", data=mAP95_s, label="Small mAP50-95")
ax2.plot("epoch", "m_mAP", data=mAP_m, label="Medium mAP")
ax2.plot("epoch", "m_mAP95", data=mAP95_m, label="Medium mAP50-95")
ax2.plot("epoch", "l_mAP", data=mAP_l, label="Large mAP")
ax2.plot("epoch", "l_mAP95", data=mAP95_l, label="Large mAP50-95")
ax2.plot("epoch", "x_mAP", data=mAP_xl, label="XL mAP")
ax2.plot("epoch", "x_mAP95", data=mAP95_xl, label="XL mAP50-95")
ax2.set_ylabel("mAP", fontsize=14, fontweight="bold")
ax2.legend(loc="best", fontsize=14)

# Add label 'b' to the second plot
ax2.text(
    -0.15,
    1.05,
    "b",
    transform=ax2.transAxes,
    fontsize=16,
    fontweight="bold",
)
ax2.tick_params(axis="both", which="major", labelsize=16)

# Set the layout
fig.supxlabel("Number of train iterations", fontsize=14, fontweight="bold")

# Save the figure
plt.savefig(
    "metrics_all.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
