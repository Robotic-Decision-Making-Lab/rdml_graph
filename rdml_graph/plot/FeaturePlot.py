# Copyright 2020 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# FeaturePlot.py
# Written Ian Rankin April 2019
#
## @package FeaturePlot
# A set of function for plotting homotopy feature points.

import numpy as np
import matplotlib.pyplot as plt

## plotFeatureNodes
# This function plots a list of feature nodes and labels them on the figure.
# @param nodes - list of nodes to plot.
def plotFeatureNodes(nodes, dashlength = 40, color='black', fontsize=12, zorder=10, annotate=True):
    if not isinstance(nodes, list):
        nodes = [nodes]

    xPts = []
    yPts = []
    for n in nodes:
        #plt.text(n.pt[0], n.pt[1], n.name, withdash=True, \
        #        dashdirection = 0, \
        #        dashlength = dashlength,\
        #        rotation = 0, \
        #        dashrotation = 0,
        #        dashpush = 10)
        if annotate:
            plt.annotate(n.name, (n.pt[0], n.pt[1]), xytext=(n.pt[0] - 0.25, n.pt[1] + 0.35), color=color, fontsize=fontsize, zorder=zorder)

        xPts.append(n.pt[0])
        yPts.append(n.pt[1])

    plt.scatter(xPts, yPts, color='black', zorder=zorder)
