import matplotlib.pyplot as plt
import pandas as pd

# load the data
mAP = pd.read_csv("mAP_nano.csv", usecols=[0, 1])
mAP95 = pd.read_csv("mAP_nano.csv", usecols=[0, 2])


# multi line-plot
fig, ax = plt.subplots(layout="constrained")
ax.plot("epoch", "n_mAP", data=mAP, label="mAP")
ax.plot("epoch", "n_mAP95", data=mAP95, label="mAP50-95")


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
