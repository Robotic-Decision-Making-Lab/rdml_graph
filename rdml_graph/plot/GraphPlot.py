# GraphPlot.py
# Written Ian Rankin November 2019
#
# This file contains code to plot different paths, and plot them.

import Node
import Path
import numpy as np
import matplotlib.pyplot as plt


# plot2DGeoGraph
# This assumes that the graph is given a geometric graph
# with 2d points, if it is not, this function may fail.
# This function is not effcient
# @param G - list of all nodes in graph
# @param color - the color of edges
def plot2DGeoGraph(G, color='blue'):
    for n in G:
        plotEdgesFromNode(n, color)


def plotEdgesFromNode(n, color='blue'):
    i = 0
    x = np.zeros(len(n.e)*2)
    y = np.zeros(len(n.e)*2)
    for e in n.e:
        x[i] = n.pt[0]
        y[i] = n.pt[1]
        i += 1
        x[i] = e.c.pt[0]
        y[i] = e.c.pt[1]
        i += 1
    plt.plot(x,y, color=color)
