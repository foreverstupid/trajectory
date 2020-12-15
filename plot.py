#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys

ref = f'''
    Script for data plotting.
    Usage:
        {__file__} <pref> [<dims>]

    <pref>:
        Data files prefix.

    <dims>:
        The count of checked dimensions for GPM.
'''

if (len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help"):
    print(ref)
    sys.exit(0)

pref = sys.argv[1]

if (len(sys.argv) > 2):
    dim = int(sys.argv[1])
else:
    dim = 6



fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(1, dim + 1):
    data = np.genfromtxt(f"{pref}_{i}.ssv", delimiter=" ", names=["x", "y"])
    ax.set_xlabel("$\ln l$")
    ax.set_ylabel("$\ln C(l)$")
    ax.plot(data["x"], data["y"])

fig.savefig(f"{pref}.png")
