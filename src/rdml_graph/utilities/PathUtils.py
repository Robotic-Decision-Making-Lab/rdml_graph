# Copyright 2021 Ian Rankin
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
# @package PathUtils.py
# Written Ian Rankin April 2021
#
# A set of function for handling paths


import numpy as np
from rdml_graph.core import Node
from rdml_graph.homotopy import HNode


## getWaypoints
# get waypoints from a list of HNodes.
# @param path - a list of homotopy nodes.
#
# @return 2d numpy array of waypoints, (n x 2)
def getWaypoints(path):
    if len(path) < 1:
        return np.empty((0, 2))
    elif isinstance(path[0], HNode):
        return getWaypointsHomotopy(path)

    pts = np.empty((len(path), 2))
    for i, n in enumerate(path):
        pts[i] = n.pt

    return pts


## getWaypoints
# get waypoints from a list of HNodes.
# @param path - a list of homotopy nodes.
#
# @return 2d numpy array of waypoints, (n x 2)
def getWaypointsHomotopy(path):
    if isinstance(path, HNode):
        path = path.get_parent_path()


    pts = np.empty((len(path), 2))
    for i, homotopy in enumerate(path):
        pts[i] = homotopy.node.pt

    return pts
