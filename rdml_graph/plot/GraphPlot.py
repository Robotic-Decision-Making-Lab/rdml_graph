# GraphPlot.py
# Written Ian Rankin November 2019
#
# This file contains code to plot different paths, and plot them.

from ..core import Node
from ..core import Edge
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

# plot2DGeoPath
# plot a geometric path given as a list of geometric nodes
# @param path - a list of GeometricNode(s) in order.
# @param color - the color of the path when plotted.
def plot2DGeoPath(path, color='red'):
    points = np.empty((len(path), path[0].pt.shape[0]))

    for i in range(len(path)):
        points[i] = path[i].pt

    plt.plot(points[:,0], points[:,1], color=color)

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
