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

## @package Node
# Node.py
# Written Ian Rankin October 2019 - edited Feb 2020
#
# A generic node structure for a graph. Can be extended to include more
# information about the node.

from rdml_graph.core import State
from rdml_graph.core import Edge

# directional graph
from graphviz import Digraph
import numpy as np


## Node class for a node of a graph structure.
#
# The node class is the basic data structure to handle nodes of a graph.
# This can be extended to include a wide variety of subclasses.
class Node(State):
    ## constructor
    # @param id - the integer the Node repersents. (int)
    def __init__(self, id):
        self.e = []
        self.id = id

    ## @var e
    # A list of Edge objects
    ## @var id
    # The unique id of the node (int)

    # @overide
    ## successor function for State
    # @return [(child, cost), ...]
    def successor(self):
        return [(edge.c, edge.getCost()) for edge in self.e]

    ## a function to allow adding an edge to the node.
    # @param edge - the input edge to add (int)
    def addEdge(self, edge):
        self.e.append(edge)

    ## returns a list of edges
    def getEdges(self):
        return self.e

    ## checks if there is a connection to a node with the otherID
    # @param otherID  - the id of the other node to check.
    def checkConnection(self, otherID):
        for edge in self.e:
            if edge.c.id == otherID:
                return edge
        return None

    ## returns a short description of the label of the node.
    # shorter than the description described by str(self)
    def getLabel(self, data=None):
        return self.id

    ############### operator overloading

    ## == operator
    # Only looks at the id's to check if it is the same node.
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    ## != operator
    # returns inverse of equals sign.
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    ## str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(self.id) + ', edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result
#######################################################

## TreeNode class
# This class extends node with an additional pointer to it's parent for easy traversal.
class TreeNode(Node):
    ## constructor
    # @param id - the integer that is the 'index' of the node.
    # @param parent_edge - the parent node (no edge given)
    def __init__(self, id, parent):
        super(TreeNode, self).__init__(id)
        self.parent = parent
    ## @var parent
    # the parent Node (no edge)


    ## Designed to be overloaded for checking if it is equals in a DFS.
    def dfs_equals(self, a, b):
        return a == b

    ## getRevPath
    # This function works its way to up the root of the tree
    # to return the list of all states in path.
    # @return a list of nodes along the tree to root [0] is the current node.
    def getRevPath(self):
        if self.parent is None:
            return [self]

        return [self] + self.parent.getRevPath()

    ## getPath
    # This function works its way to up the search tree to the root node
    # to return the list of all states in path.
    # @return a list of nodes along the tree to root [0] is the root
    def getPath(self):
        if self.parent is None:
            return [self]

        return self.parent.getPath() + [self]

    # getTreeStats
    # This function returns various statistics about the current tree
    def getTreeStats(self, max_depth = 0):
        node_balance = []
        max_depth = max_depth
        new_max_depth = max_depth
        num_branches = 0

        for edge in self.e:
            if isinstance(edge.c, TreeNode):
                stats = edge.c.getTreeStats(max_depth+1)
            else:
                stats = {}
                stats['max_depth'] = max_depth+1
                stats['node_balance'] = [1]
                stats['num_branches'] = 1

            if stats['max_depth'] > new_max_depth:
                new_max_depth = stats['max_depth']

            node_balance.append(sum(stats['node_balance']))
            num_branches += stats['num_branches']

        stats = {}
        stats['max_depth'] = new_max_depth
        stats['node_balance'] = node_balance
        stats['num_branches'] = num_branches
        return stats



    ## depth first search
    # returns the full path to the output location.
    # @param n - the input node
    #
    # @return list of nodes from the leaf
    def dfs(self, n):
        # base-case
        if self.dfs_equals(self, n):
            return [n]
        else:
            for edge in self.e:
                if self.dfs_equals(edge.c, n):
                    return [self, edge.c]
                elif isinstance(edge.c, TreeNode):
                    ret = edge.c.dfs(n)
                    if ret is not None:
                        return [self] + ret

            # No object found, return None
            return None

    ## depth first search
    # returns the full path to the output location.
    # param n - the input node
    #
    # @return - list of nodes from the leaf
    def dfs_edge(self, n):
        # base-case
        if self.dfs_equals(self, n):
            return []
        else:
            for edge in self.e:
                if self.dfs_equals(edge.c, n):
                    return [edge]
                elif isinstance(edge.c, TreeNode):
                    ret = edge.c.dfs_edge(n)
                    if ret is not None:
                        return [edge] + ret

            # No object found, return None
            return None


    ## A set of code to get visualization code for a tree.
    # This uses the graphviz python library to generate a Digraph object for tree.
    # @oaram labels - boolean for if the tree should inlcude lables.
    # @param t - a Digraph object to start with (leave if creating a new viz)
    #
    # @return - Digraph object ( t.view() ) called after will show the tree.
    def get_viz(self, labels=False, t=None, data=None):
        if t is None:
            t = Digraph('T')

        if labels:
            label = self.get_plot_label(data=data)
            t.node(str(self.id), label)
        else:
            t.node(str(self.id), '')

        if self.parent is not None:
            t.edge(str(self.parent.id), str(self.id))

        # recursivly call sub calls
        for e in self.e:
            if isinstance(e.c, TreeNode):
                t = e.c.get_viz(labels, t, data)
        return t


    ## Designed to be overloaded for labels in visualization
    def get_plot_label(self, data=None):
        return str(self.id)

    def __str__(self):
        result = 'node(id='+ str(self.id) + ', parent=' + str(self.parent) + ', edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result


## GeometricNode that includes a geometric point as part of the
# node as well as the graph structure.
class GeometricNode(Node):
    ## Constructor
    # @param id - the integer that is the 'index' of the node.
    # @param pt - numpy array representing spatial point.
    def __init__(self, id, pt):
        super(GeometricNode, self).__init__(id)
        self.pt = pt
    ## @var pt
    # The numpy array representing the spatial point.

    ## str(self) operator
    # Returns a quick human readable string
    def __str__(self):
        result = 'node(id='+ str(self.id) + ', pt='+ str(self.pt) +' edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result
