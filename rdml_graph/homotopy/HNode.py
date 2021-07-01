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
# HNode.py
# Written Ian Rankin - February 2020
#
# Homotopy augumented node.
# This is a set of nodes built on-top of a standard
# Node and graph, but has a different successor function
#
# This is designed to work with either Homotopy or Homology signatures.
# This makes code re-use easier or more straight forward.
#
# Based on work by following paper:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#

from rdml_graph.core import State
from rdml_graph.core import Node
from rdml_graph.core import TreeNode
from rdml_graph.core import Edge
from rdml_graph.homotopy import HSignature

#from numba import njit
import numpy as np

import pdb


class HNode(TreeNode):
    # constructor
    # @param node - the input Node for h-augmented graph. (Either homotopy or homology)
    # @param h_sign - the input H signature.
    # @param parent - [optional] the parent HNode
    # @param root - [optional] the root node of the homotopy graph.
    def __init__(self, n, h_sign, parent=None, root=None):
        super(HNode, self).__init__(-2, parent)
        self.node = n
        self.h_sign = h_sign
        self.root = root
        self.e = None

    # successor function for Homotopy node.
    def successor(self):
        if self.e is not None:
            return [(e.c, e.getCost()) for e in self.e]

        self.e = []
        # self.e = [Edge(self, HNode(n=edge.c, h_sign=self.h_sign.copy(), parent=self, root=self.root), edge.getCost()) \
        #     for edge in self.node.e if (self.h_sign.copy()).edge_cross(edge)]
        for edge in self.node.e:
            newHSign = self.h_sign.copy()
            goodHSign = newHSign.edge_cross(edge)

            if goodHSign:
                succ = HNode(n=edge.c, h_sign=newHSign,\
                                parent=self, root=self.root)
                self.e.append(Edge(self, succ, edge.getCost()))
        #pdb.set_trace()
        return [(e.c, e.getCost()) for e in self.e]

    def get_parent_path(self):
        # base case
        if self.parent is None:
            return [self]
        else:
            return self.parent.get_parent_path() + [self]


    ################## operator overloading

    # ==
    # equals checks other is the same class and then checks node and h-signature
    # for equality.
    def __eq__(self, other):
        if not isinstance(other, HNode):
            return False
        return self.node == other.node and self.h_sign == other.h_sign and \
                (self.root is None or other.root is None or self.root == other.root)

    # !=
    # opposite of equals
    def __ne__(self, other):
        return not self == other

    # str()
    # prints out info about the currnet Homotopy Node
    def __str__(self):
        return 'HNode(h-sign='+ str(self.h_sign) +', n=' + str(self.node.id) + ')'

    # hash function overload
    # This hash takes into account both the node hash (should be defined),
    # and the h signatures hash (also defined).
    # parent edge is not considered.
    ###### THIS is actually important for SEARCHES as it defines what is considered
    # already explored.
    def __hash__(self):
        return hash((self.node, self.h_sign))






































    #
