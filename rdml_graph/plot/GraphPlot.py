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
# GraphPlot.py
# Written Ian Rankin November 2019
#
# This file contains code to plot different paths, and plot them.

from rdml_graph.core import Node
from rdml_graph.core import Edge
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


# CODE From stack overflow: https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit
class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        #self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()

# plot2DPath
# This function plots a 2D path with arrows showing directionality
# @param pts - a list of 2D points along the path
# @param color - the color of the path when plotted.
# @param head_width - the width of the arrow head
def plot2DPath(path, color='red', line_style='solid', head_width=0.75, label='', radius=None):
    line = None
    for i in range(1, len(path)):
        diff = path[i] - path[i-1]
        line = plt.arrow(path[i-1][0], path[i-1][1], diff[0], diff[1], \
                length_includes_head=True, \
                head_width=head_width, color=color,\
                label=label)

    if radius is not None:
        #plt.plot(path[:,0], path[:,1], color=color, alpha=0.3, \
        #    linewidth=radius*2, solid_capstyle='round')
        l = data_linewidth_plot(path[:,0], path[:,1], color=color, alpha=0.3, \
                               linewidth=radius, solid_capstyle='round')
        pass


    return line

# plot2DGeoPath
# plot a geometric path given as a list of geometric nodes
# @param path - a list of GeometricNode(s) in order.
# @param color - the color of the path when plotted.
def plot2DGeoPath(path, color='red'):
    if len(path) <= 0:
        return
    points = np.empty((len(path), path[0].pt.shape[0]))

    for i in range(len(path)):
        points[i] = path[i].pt

    plt.plot(points[:,0], points[:,1], color=color)

# plotHomotopyPath
# plot a geometric path given as a list of geometric nodes
# @param path - a list of GeometricNode(s) in order.
# @param color - the color of the path when plotted.
def plotHomotopyPath(path, color='red'):
    if path is None or len(path) <= 0:
        return
    points = np.empty((len(path), path[0].node.pt.shape[0]))

    for i in range(len(path)):
        points[i] = path[i].node.pt

    plt.plot(points[:,0], points[:,1], color=color)

# plotEdgesFromNode
# This function is given a node and plots edges coming from that node.
# @param n - the node
# @param color - the color of the edges.
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
