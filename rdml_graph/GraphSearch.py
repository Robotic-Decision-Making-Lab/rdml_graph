# AStar.py
# Written Ian Rankin February 2020 - Modified from code written October 2019
#
# The AStar algorithm written using the SearchState class

from rdml_graph import SearchState
from rdml_graph import State
import heapq


# defualt huerestic function for AStar.
# it should work for any type of state, but when using the algorithm degrades
# to simply Dijkstra's algorithm rather than A*.
# @param x - the graph input path.
def default_h(x, data = None):
    return 0.0

# AStar
# A generic implementation of the AStar algorithm.
# An optimal graph search algorithm.
# @param start - the start state of the search
# @param g - a goal function to determine if the passed, state is in the goal set.
def AStar(start, g, h = default_h, data = None):
    pass



def BFS(start):
    pass
