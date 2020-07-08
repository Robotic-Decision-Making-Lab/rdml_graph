# FeatureNode.py
# Written Ian Rankin - March 2020
#
# Feature Node a Node of graph which also contains a feature description
# Contains a semantic name which precisely describes the feature, as well
# a set of keywords which describe the node.
# Ex. name = "Shaw Island"
#     keywords = {"Island", "Shaw", "Isle", "Atoll"}
#
#

from ..core import GeometricNode
from . import HomotopyNode
import copy

import pdb

class FeatureNode(GeometricNode):
    # constructor
    # @param id - an integer which describes the node.
    # @param name - the name of the feature it represents.
    # @param keywords - a list or set of keywords describing the feature (can be empty)
    def __init__(self, id, name, pt=None, keywords={}):
        super(FeatureNode, self).__init__(id, pt)

        self.name = name
        self.keywords=set(keywords)

    def __str__(self):
        result = 'node(id='+ str(self.id) + ', name=' + str(self.name) + \
                ', keywords=' + str(self.keywords) + ', pt='+ str(self.pt) + \
                ' edges={'
        for edge in self.e:
            result += str(edge)+','
        result += '})'
        return result


# A state that incapsulates the set of states of Homotopy and features
class HomotopyFeatureState(HomotopyNode):
    # constructor
    # @param node - the input Node for homotopy graph.
    # @param h_sign - the input H signature.
    # @param parent - [optional] the edge from the parent HomotopyNode
    # @param root - [optional] the root node of the homotopy graph.
    # @param names - a set of names along path
    # @param keywords - a set of keywords along path.
    def __init__(self, n, h_sign, parentEdge=None, root=None, names=frozenset(), neededNames=frozenset()):
        super(HomotopyFeatureState, self).__init__(n, h_sign, parentEdge, root)

        self.names = names
        self.neededNames = neededNames

    # successor function for Homotopy node.
    def successor(self):
        result = []
        for edge in self.node.e:
            newHSign = self.h_sign.copy()
            goodHSign = newHSign.edgeCross(edge)

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

    # hash function overload
    # This hash takes into account both the node hash (should be defined),
    # and the h signatures hash (also defined).
    # parent edge is not considered.
    ###### THIS is actually important for SEARCHES as it defines what is considered
    # already explored.
    def __hash__(self):
        return hash((self.node, self.h_sign, self.names))



#
# # A state that incapsulates the set of states of Homotopy and features
# class HomotopyFeatureState(HomotopyNode):
#     # constructor
#     # @param node - the input Node for homotopy graph.
#     # @param h_sign - the input H signature.
#     # @param parent - [optional] the edge from the parent HomotopyNode
#     # @param root - [optional] the root node of the homotopy graph.
#     # @param names - a set of names along path
#     # @param keywords - a set of keywords along path.
#     def __init__(self, n, h_sign, parentEdge=None, root=None, names=frozenset(), keywords=frozenset()):
#         super(HomotopyFeatureState, self).__init__(n, h_sign, parentEdge, root)
#
#         self.names = names
#         self.keywords = keywords
#
#     # successor function for Homotopy node.
#     def successor(self):
#         result = []
#         for edge in self.node.e:
#             newHSign = self.h_sign.copy()
#             goodHSign = newHSign.edgeCross(edge)
#
#             if goodHSign:
#                 newNames = None
#                 newKeywords = None
#                 if isinstance(edge.c, FeatureNode):
#                     newNames = frozenset(self.names | {edge.c.name})
#                     newKeywords = frozenset(self.keywords | edge.c.keywords)
#                 else:
#                     newNames = frozenset(self.names)
#                     newKeywords = frozenset(self.keywords)
#
#                 succ = HomotopyFeatureState(n=edge.c, h_sign=newHSign,\
#                             names=newNames, keywords=newKeywords, parentEdge=edge,\
#                             root=self.root)
#                 result += [(succ, edge.getCost())]
#         #pdb.set_trace()
#         return result
#
#     # hash function overload
#     # This hash takes into account both the node hash (should be defined),
#     # and the h signatures hash (also defined).
#     # parent edge is not considered.
#     ###### THIS is actually important for SEARCHES as it defines what is considered
#     # already explored.
#     def __hash__(self):
#         return hash((self.node, self.h_sign, self.root, self.names, self.keywords))
