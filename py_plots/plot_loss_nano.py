import matplotlib.pyplot as plt
import pandas as pd

# load the data
box = pd.read_csv("py_plots/loss_nano.csv", usecols=[0, 1])
dfl = pd.read_csv("py_plots/loss_nano.csv", usecols=[0, 2])


# multi line-plot
fig, ax = plt.subplots(layout="constrained")
ax.plot("epoch", "n_dfl", data=dfl, label="DFL (distribution focal loss)")
ax.plot("epoch", "n_box", data=box,
        label="$L_{CIoU}$ (complete intersection over union loss)")


ax.set_xlabel(
    "Number of train iterations", fontdict={"fontsize": 12, "fontweight": "bold"}
)
ax.tick_params(axis="both", labelsize=13)
ax.set_ylabel("Loss", fontdict={"fontsize": 12, "fontweight": "bold"})
# ax.set_title("Loss value of the detection model", fontdict={"fontsize": 18})
ax.legend(loc="best", fontsize=12)

plt.savefig(
    "loss_nano.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
