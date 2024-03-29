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
## @package FeatureNode.py
# Written Ian Rankin - March 2020
#
# Feature Node a Node of graph which also contains a feature description
# Contains a semantic name which precisely describes the feature, as well
# a set of keywords which describe the node.
# Ex. name = "Shaw Island"
#     keywords = {"Island", "Shaw", "Isle", "Atoll"}
#
#

from rdml_graph.core import GeometricNode
from rdml_graph.homotopy import HNode
import copy

import pdb

class FeatureNode(GeometricNode):
    ## constructor
    # @param id - an integer which describes the node.
    # @param name - the name of the feature it represents.
    # @param keywords - a list or set of keywords describing the feature (can be empty)
    def __init__(self, id, name, pt=None, keywords={}, obs=None):
        super(FeatureNode, self).__init__(id, pt)

        self.name = name
        self.keywords=set(keywords)
        self.obs = obs

    def __str__(self):
        result = 'node(id='+ str(self.id) + ', name=' + str(self.name) + \
                ', keywords=' + str(self.keywords) + ', pt='+ str(self.pt) + \
                ' edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result


## A state that incapsulates the set of states of Homotopy and features
class HomotopyFeatureState(HNode):
    ## constructor
    # @param node - the input Node for homotopy graph.
    # @param h_sign - the input H signature.
    # @param parent - [optional] the edge from the parent HNode
    # @param root - [optional] the root node of the homotopy graph.
    # @param names - a set of names along path
    # @param keywords - a set of keywords along path.
    def __init__(self, n, h_sign, parentEdge=None, root=None, names=frozenset(), neededNames=frozenset()):
        super(HomotopyFeatureState, self).__init__(n, h_sign, parentEdge, root)

        self.names = names
        self.neededNames = neededNames

    ## successor function for Homotopy node.
    def successor(self):
        result = []
        for edge in self.node.e:
            newHSign = self.h_sign.copy()
            goodHSign = newHSign.edge_cross(edge)

            if goodHSign:
                newNames = None
                newKeywords = None
                if isinstance(edge.c, FeatureNode) and edge.c.name.lower() in self.neededNames:
                    newNames = frozenset(self.names | {edge.c.name.lower()})
                else:
                    newNames = frozenset(self.names)

                succ = HomotopyFeatureState(n=edge.c, h_sign=newHSign,\
                            names=newNames, neededNames=self.neededNames, parentEdge=edge,\
                            root=self.root)
                result += [(succ, edge.getCost())]
        #pdb.set_trace()
        return result

    ## hash function overload
    # This hash takes into account both the node hash (should be defined),
    # and the h signatures hash (also defined).
    # parent edge is not considered.
    ###### THIS is actually important for SEARCHES as it defines what is considered
    # already explored.
    def __hash__(self):
        return hash((self.node, self.h_sign, self.names))
