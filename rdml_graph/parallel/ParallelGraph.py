# ParallelGraph.py
# Written Ian Rankin - May 2021
#
# Basic functions to handle graphs for massively parallel interactions.

#from numba import jit
#from numba import cuda
import numpy as np
import math


def get_edges(n):
    idxs = np.empty(len(n.e))
    costs = np.empty(len(n.e))

    for i, edge in enumerate(n.e):
        idxs[i] = edge.c.id
        costs[i] = edge.cost

    return idxs, costs

# @param G - list of nodes (assume nodes have sequential id's)
#
# @return - numpy array of edges, costs, number of edges
def get_connection_graph(G):
    # find max number of edges
    max_num_edges = 0
    for n in G:
        if len(n.e) > max_num_edges:
            max_num_edges = len(n.e)

    A       = np.ones((len(G), max_num_edges), dtype=np.int32) * -1
    costs   = np.ones((len(G), max_num_edges), dtype=np.float32) * -1
    num_edges = np.empty(len(G))

    for n in G:
        edges, n_costs = get_edges(n)

        A[n.id, :len(edges)] = edges
        costs[n.id, :len(edges)] = n_costs
        num_edges[n.id] = len(edges)

    return A, costs, num_edges
