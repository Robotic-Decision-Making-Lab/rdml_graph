# FeaturePlot.py
# Written Ian Rankin April 2019
#
# A set of function for plotting homotopy feature points.

import numpy as np
import matplotlib.pyplot as plt

# plotFeatureNodes
# This function plots a list of feature nodes and labels them on the figure.
# @param nodes - list of nodes to plot.
def plotFeatureNodes(nodes, dashlength = 40):
    if not isinstance(nodes, list):
        nodes = [nodes]

    xPts = []
    yPts = []
    for n in nodes:
        plt.text(n.pt[0], n.pt[1], n.name, withdash=True, \
                dashdirection = 0, \
                dashlength = dashlength,\
                rotation = 0, \
                dashrotation = 0,
                dashpush = 10)
        xPts.append(n.pt[0])
        yPts.append(n.pt[1])

    plt.scatter(xPts, yPts)
