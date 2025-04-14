import matplotlib.pyplot as plt
import pandas as pd

# load the data
mAP = pd.read_csv("py_plots/mAP_nano.csv", usecols=[0, 1])
mAP95 = pd.read_csv("py_plots/mAP_nano.csv", usecols=[0, 2])

# multi line-plot
fig, ax = plt.subplots(layout="constrained")
ax.plot("epoch", "n_mAP", data=mAP, label="mAP")
ax.plot("epoch", "n_mAP95", data=mAP95, label="mAP50-95")

# Highlight the best model at epoch 105
best_epoch = 105
best_mAP = mAP[mAP['epoch'] == best_epoch]['n_mAP'].values[0]
best_mAP95 = mAP95[mAP95['epoch'] == best_epoch]['n_mAP95'].values[0]

# Add a vertical dashed line at the best epoch instead of markers
ax.axvline(x=best_epoch, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

# Add annotation for the best model
ax.annotate(f'Best model (epoch {best_epoch})',
            xy=(best_epoch, best_mAP), 
            xytext=(best_epoch+10, best_mAP+0.02),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10)

ax.set_xlabel(
    "Number of train iterations", fontdict={"fontsize": 12, "fontweight": "bold"}
)
ax.tick_params(axis="both", labelsize=13)
ax.set_ylabel(
    "mAP (mean average precision)", fontdict={"fontsize": 12, "fontweight": "bold"}
)
# ax.set_title("mAP value of the detection model", fontdict={"fontsize": 18})
ax.legend(loc="best", fontsize=12)

plt.savefig(
    "mAP_nano.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
