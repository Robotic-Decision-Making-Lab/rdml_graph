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

# @package AStar.py
# Written Ian Rankin February 2020 - Modified from code written October 2019
#
# The AStar algorithm written using the SearchState class

from rdml_graph.core import SearchState
from rdml_graph.core import State
import sys
# For the priority queue used by the AStar algorith.
import heapq
# For queue
import numpy as np

import pdb

## defualt huerestic function for AStar.
# it should work for any type of state, but when using the algorithm degrades
# to simply Dijkstra's algorithm rather than A*.
# @param x - the graph input path.
def default_h(x, data = None, goal=None):
    return 0.0

## A simple euclidean distance huerestic for the AStar algorithm
# I doubt this does particulary much to speed on computation if any at all.
def h_euclidean(n, data, goal):
    return np.linalg.norm(n.node.pt - goal[0].pt)

## graphGoalCheck
# A basic graph checker looking for a particular node to be the same.
# @param n - the node to check
# @param data - some set of input data
def graph_goal_check(n, data, goal):
    return n == goal

## pass_all
# goal check that allows all through
def pass_all(n, data, goal):
    return True




## AStar
# A generic implementation of the AStar algorithm.
# An optimal graph search algorithm.
# If looking for shortest path to all nodes, see Dijkstra's algorithm.
# REQUIRED
# @param start - the start state of the search
# OPTIONAL
# Functions g and h have input types of (state, data, goal)
# @param g - a goal function to determine if the passed, state is in the goal set.
# @param h - a heuristic function for the AStar search needs type (state, data, goal)
# @param data - a potential set of input data for huerestics and goal states.
# @param goal - a potential set of goal data (REQUIRED by default)
# @param out_tree - if true, output tree, otherwise do not output a tree.
# @param keepEdges - [opt] if true, keep the edges in the path.
# @param keepNodes - [opt] if true, keep the nodes in the path.
#
# @returns - list, cost
#   an optimal list states to the goal state. - if no path return empty list and infinte cost.
# [first state, ---, goal state]
def AStar(start, g=graph_goal_check, h = default_h, data = None, goal=None, \
            output_tree=False, keepEdges=False, keepNodes=True):
    startState = SearchState(start, hCost=h(start, data, goal), id=0)
    frontier = [startState]
    explored = set()

    i = 0

    cur_id = 1
    while len(frontier) > 0:
        #pdb.set_trace()

        i += 1

        # get current state to explore
        cur = heapq.heappop(frontier)

        if cur.state not in explored:
            # check if the current state is in the goal state.
            if g(cur.state, data, goal):
                if output_tree:
                    return cur.getPath(keepEdges=keepEdges, keepNodes=keepNodes), \
                                    cur.rCost, startState
                else:
                    return cur.getPath(keepEdges=keepEdges, keepNodes=keepNodes), \
                                    cur.rCost

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
    print(i)
    # End of while, no solution found.
    if output_tree:
        return [], float('inf'), startState
    else:
        return [], float('inf')


## dijkstra's algorithm (All nodes)
# This is dijkstra's algorithm ran to find the shortest path to all reachable
# nodes of a graph from the start location.
# See any algorithms book for description of dijkstra's algorithm.
# Very similar to the above AStar algorithm without being single query, and
# without a huerestic function.
# @param start - the start location for dijkstra's algorithm (must be a State class)
#
# @return - a dictionary of every SearchState in the tree. (key is state)
def dijkstra(start):
    startState = SearchState(start)
    frontier = [startState]
    explored = {}

    while len(frontier) > 0:
        # get current state to explore
        cur = heapq.heappop(frontier)

        if cur.state not in explored:

            # add state to dict of explored states
            explored[cur.state] = cur

            # get list of successors
            successors = cur.successor()

            # add all successors to frontier
            for succ in successors:
                # check to make sure state hasn't already been explored.
                if succ.state not in explored:
                    heapq.heappush(frontier, succ)

    # End of while, return all found paths.
    return explored


# DFS
# Depth first search
# A depth first search which has a budget which acts as a dynamic iterative
# deepening search (IDS). This is a generic function for performing a search
# of a tree structure.
def DFS():
    pass
    # Not implemented





## BFS
# Breadth First Search algorithm. This has an optional budget which restricts
# expansion beyond the budget given the cost of the state function returned
# REQUIRED
# @param start - the starting
# OPTIONAL
# @param budget - the budget of the search, if ignored, no cost budget is given.
# @param g - a goal condition that must be met in order to be in the returned states.
#                 g(n, data, goal)
# @param data - input data for the goal function if required.
# @param goal - any goal data required by g function.
# @param keepEdges - [opt] if true, keep the edges in the path.
# @param keepNodes - [opt] if true, keep the nodes in the path.
#
# @return - list of structs of [(path, cost),...] or [([n1,n2,n3,...], cost), ...]
def BFS(start, budget=float('inf'), g=pass_all, data=None, goal=None, keepEdges=False, keepNodes=True):
    startState = SearchState(start)
    #frontier = queue.Queue()
    #frontier.put(startState)
    frontier = [startState]
    explored = set()

    endStates = []

    while len(frontier) > 0:
        cur = frontier.pop(0)

        if cur.state not in explored:
            # add state to set of explored states
            explored.add(cur.state)

            # check for end cases
            if cur.cost() >= budget and g(cur.state, data, goal):
                endStates.append(cur)
            else:
                # get list of successors
                successors = cur.successor()
                # add all successors to frontier that have not been explored
                anyExplored = False
                for succ in successors:
                    # check to make sure state hasn't already been explored.
                    if succ.state not in explored:
                        anyExplored = True
                        frontier.append(succ)
                if anyExplored == True and g(cur.state, data, goal):
                    endStates.append(cur)

    return [(s.getPath(keepEdges=keepEdges, keepNodes=keepNodes), s.rCost) for s in endStates]
