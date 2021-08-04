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
## @package HNode.py
# Written Ian Rankin - February 2020
#

#

from rdml_graph.core import State
from rdml_graph.core import Node
from rdml_graph.core import TreeNode
from rdml_graph.core import Edge
from rdml_graph.homotopy import HSignature

#from numba import njit
from rdml_graph.parallel import dev_rand_int
from numba import cuda
import numpy as np

import pdb

## # Homotopy or homology augumented node.
# This is a set of nodes built on-top of a standard
# Node and graph, but has a different successor function
#
# This is designed to work with either Homotopy or Homology signatures.
# This makes code re-use easier or more straight forward.
#
# Based on work by following paper:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
class HNode(TreeNode):
    ## constructor
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

    ## successor function for Homotopy node.
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




    ################## operator overloading

    ## ==
    # equals checks other is the same class and then checks node and h-signature
    # for equality.
    def __eq__(self, other):
        if not isinstance(other, HNode):
            return False
        return self.node == other.node and self.h_sign == other.h_sign and \
                (self.root is None or other.root is None or self.root == other.root)

    ## !=
    # opposite of equals
    def __ne__(self, other):
        return not self == other

    ## str()
    # prints out info about the currnet Homotopy Node
    def __str__(self):
        return 'HNode(h-sign='+ str(self.h_sign) +', n=' + str(self.node.id) + ')'

    ## hash function overload
    # This hash takes into account both the node hash (should be defined),
    # and the h signatures hash (also defined).
    # parent edge is not considered.
    ###### THIS is actually important for SEARCHES as it defines what is considered
    # already explored.
    def __hash__(self):
        return hash((self.node, self.h_sign))



    ################ parallel speedup code

    @staticmethod
    def para_get_rep(sequence):
        path = [hn.node.pt for hn in sequence]
        return np.array(path, dtype=np.float32)


    @staticmethod
    def para_propogate(parallel_data, states, length_states, budget):
        threads_per_block = 64
        blocks = len(length_states) % threads_per_block

        kern_rand_travesal_and_path[blocks, threads_per_block](\
                        A = parallel_data[0], \
                        costs= parallel_data[1], \
                        num_edges= parallel_data[2], \
                        paths=states,\
                        start_locs= ,\
                        cur_lengths=length_states,\
                        points=parallel_data[4],\
                        budget=budget, \
                        rng_states = parallel_data[5])
        return states, cur_lengths




## random traversal of a graph to generate the list of paths on a graph
# @param A - the input connection graph (in the form of list of edges) (n x edges)
# @param costs - the cost with a parallel shape to A (n x edges)
# @param num_edges - the list of the number of edges (n,)
# @param paths[out] - the list of paths (n, len_paths)
# @param start_locs - the start indicies of traversal
# @param budgets - list of budgets allowed for the travesal (n,)
# @param rng_states - the rng_states for each core (n, )
@cuda.jit
def kern_rand_travesal_and_path(A, costs, num_edges, paths, start_locs, cur_lengths, \
        points, budget, rng_states):
    n = cuda.grid(1)

    if n < start_locs.shape[0]:
        # traverse
        length = cur_lengths[n]
        cur_node = start_locs[n]

        for i in range(paths.shape[1]):
            if num_edges[cur_node] == 0:
                break
            rand_num = dev_rand_int(rng_states, n, 0, num_edges[cur_node])

            length += costs[cur_node, rand_num]
            cur_node = A[cur_node, rand_num]
            paths[n, cur_lengths[n], :] = points[cur_node,:]
            cur_lengths[n] += 1

            if length >= budgets[n]:
                break

































    #
