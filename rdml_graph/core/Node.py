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
# Node.py
# Written Ian Rankin October 2019 - edited Feb 2020
#
# A generic node structure for a graph. Can be extended to include more
# information about the node.

from . import State
from . import Edge

class Node(State):
    # constructor
    # @param id - the integer the Node repersents.
    def __init__(self, id):
        self.e = []
        self.id = id

    # @overide
    # successor function for State
    def successor(self):
        return [(edge.c, edge.getCost()) for edge in self.e]

    # a function to allow adding an edge to the node.
    def addEdge(self, edge):
        self.e.append(edge)

    # returns a list of edges
    def getEdges(self):
        return self.e

    # returns a short description of the label of the node.
    # shorter than the description described by str(self)
    def getLabel(self):
        return self.id

    ############### operator overloading

    # == operator
    # Only looks at the id's to check if it is the same node.
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    # != operator
    # returns inverse of equals sign.
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    # str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(self.id) + ', edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result

#######################################################
# GeometricNode that includes a geometric point as part of the
# node as well as the graph structure.
class GeometricNode(Node):
    # Constructor
    # @param id - the integer that is the 'index' of the node.
    # @param pt - numpy array repersenting spatial point.
    def __init__(self, id, pt):
        super(GeometricNode, self).__init__(id)
        self.pt = pt

    # str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(self.id) + ', pt='+ str(self.pt) +' edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result
