# AStar.py
# Written Ian Rankin February 2020 - Modified from code written October 2019
#
# The AStar algorithm written using the SearchState class

from . import SearchState
from . import State
from ..homotopy import HSignatureGoal
# For the priority queue used by the AStar algorith.
import heapq
# For queue
from collections import deque

# defualt huerestic function for AStar.
# it should work for any type of state, but when using the algorithm degrades
# to simply Dijkstra's algorithm rather than A*.
# @param x - the graph input path.
def default_h(x, data = None, goal=None):
    return 0.0

# graphGoalCheck
# A basic graph checker looking for a particular node to be the same.
# @param n - the node to check
# @param data - some set of input data
def graph_goal_check(n, data, goal):
    return n == goal

# partial_homotopy_goal_check
# Checks if the nodes are the same, and checks if the h-signatures fit the
# constraints in HSignatureGoal, which allows partial h-signature matches.
# @param n - the input node (Should be a HomotopyNode)
# @param data - generic (not used)
# @param goal - the input goal data to the search funcion.
#               MUST match (Node, HSignatureGoal type)
def partial_homotopy_goal_check(n, data, goal):
    goalNode = goal[0]
    goalH = goal[1]
    if not isinstance(goalH, HSignatureGoal):
        raise TypeError("partial_homotopy_goal_check given goal which should be (Node, HSignatureGoal)")

    return goalH.checkSign(n.h_sign) and goalNode == n.node

# AStar
# A generic implementation of the AStar algorithm.
# An optimal graph search algorithm.
# REQUIRED
# @param start - the start state of the search
# OPTIONAL
# Functions g and h have input types of (state, data, goal)
# @param g - a goal function to determine if the passed, state is in the goal set.
# @param h - a heuristic function for the AStar search needs type (state, data, goal)
# @param data - a potential set of input data for huerestics and goal states.
# @param goal - a potential set of goal data (REQUIRED by default)
#
# @returns - list, cost
#   an optimal list states to the goal state. - if no path return empty list.
# [first state, ---, goal state]
def AStar(start, g=graph_goal_check, h = default_h, data = None, goal=None):
    startState = SearchState(start, hCost=h(start, data, goal))
    frontier = [startState]
    explored = set()

    while len(frontier) > 0:
        # get current state to explore
        cur = heapq.heappop(frontier)

        if cur.state not in explored:
            # check if the current state is in the goal state.
            if g(cur.state, data, goal):
                return cur.getPath(), cur.rCost

            # add state to set of explored states
            explored.add(cur.state)

            # get list of successors
            successors = cur.successor()

            # add all successors to frontier
            for succ in successors:
                # check to make sure state hasn't already been explored.
                if succ.state not in explored:
                    # run heuristic function.
                    succ.hCost = h(succ.state, data, goal)
                    heapq.heappush(frontier, succ)
    # End of while, no solution found.
    return [], float('inf')



def BFS(start):
    pass
