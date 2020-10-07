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
# HEdge.py
# Written Ian Rankin - February 2020
#
# Homotopy augumented edge.
# This is a set of edges defining crossings given the edge.
# Slightly different from Homotopy Node which keeps a copy of the node it is
# representing, but the actual node.
#
# This edge is just an extension of an Edge with extra information to contain
# crossing information.
#
# Based on work by following paper:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#

from __future__ import absolute_import

from rdml_graph.core import Edge
import numpy as np
import sys

# if sys.version_info[0] > 2:
#     #from . import HSignature
#     from rdml_graph.homotopy import HSignature
# else:
#     #from rdml_graph.homotopy import HSignature
#     import rdml_graph.homotopy
import rdml_graph.homotopy




class HEdge(Edge):
    # constructor
    # @param parent - the parent node of the edge.
    # @param child - the child node of the edge.
    # @param num_objects - the total number of topological objects.
    # @param cost - the cost of the edge - defaults to 1
    def __init__(self, parent, child, HSign, num_objects=0, cost=1, features=None,ray_angle=np.pi/2):
        super(HEdge, self).__init__(parent, child, cost)
        # H-signature fragment (only shows crossing that can occur)

        if features is not None:
            num_objects = features.shape[0]

        self.HSign = HSign.copy()

        if features is not None:
            self.HSign.compute_line_segment(parent.pt, child.pt, features, ray_angle)



    def __str__(self):
        return 'e(p.id='+str(self.p.id)+',c.id='+str(self.c.id)+',hFrag='+str(self.HSignFrag)+',cost='+str(self.cost)+')'
