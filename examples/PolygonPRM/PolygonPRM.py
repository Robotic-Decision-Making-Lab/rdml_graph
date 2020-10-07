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
# PolygonPRM.py
# Written Ian Rankin - October 2020
#
# This example  2D PRM with obstcales and a non-rectangular environment
# A* algorithm finding the shortest point between two points on the PRM.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as geo

bounding = geo.Polygon([(-10,25), (0, -35), (10, 25)])
obs = [geo.Polygon([(-5,5), (0, -5), (10, 2)])]
map = {'bounding': bounding, 'obs': obs}

G = gr.PRM( map=map, \
            num_points=200,\
            r= 10.0, \
            sampleF=gr.sample2DPolygon, \
            collision=gr.polygonCollision)

path, cost = gr.AStar(G[0], goal = G[50])

# plot the geometric 2d graph
gr.plot2DGeoGraph(G, 'green')
gr.plot2DGeoPath(path, 'red')
plt.show()
