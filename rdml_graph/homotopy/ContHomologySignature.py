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
# HomologyNode.py
# Written Ian Rankin - February 2020
#
# Homology augumented node.
# This is a set of nodes built on-top of a standard
# Node and graph, but has a different successor function for the H2-augmented graph
# (H2 = homology here)
# Based on work by following paper:
# S. Bhattacharya, R. Ghrist, V. Kumar (2015) Persistent Homology for Path Planning
#       in uncertain environments.
#
##################################### Incomplete ############################

from rdml_graph.core import State
from rdml_graph.core import Node
from rdml_graph.core import Edge
from rdml_graph.homotopy import HSignature

import numpy as np

import pdb

# Compute the signed angle (+ = CW, - = CCW) between ps-vref and pe-vref.
# Function based on Code from Seth McCammon.
# @param ps - the starting point of the line segment (numpy array x 2)
# @param pe - the ending point of the line segment (numpy array x 2)
# @param pref - the reference point of the line segment (numpy array n x 2)
#
# @return - angle between line segments from reference point
def computeSubtendedAngleLineSegment(ps, pe, pref):
    vref_ps = ps[np.newaxis, :] - pref
    vref_pe = pe[np.newaxis, :] - pref
    return np.arctan2(vref_ps[:,1], vref_ps[:,0]) - math.arctan2(vref_pe[:,1], vref_pe[:,0])
