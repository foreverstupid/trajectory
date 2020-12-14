import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(1, 6):
    data = np.genfromtxt(f"plot_{i}.ssv", delimiter=" ", names=["x", "y"])
    ax.set_xlabel("$\ln l$")
    ax.set_ylabel("$\ln C(l)$")
    ax.plot(data["x"], data["y"])

fig.savefig("plot.png")