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
## @package MCTSHelper.py
# Written Ian Rankin February 2020
#
# A set of various helper functions for the MCTS Tree search.

import tqdm
import numpy as np
from rdml_graph.mcts import MCTSTree
from rdml_graph.mcts.ParetoFront import ParetoFront, get_pareto

import pdb

########################### Common functions for MCTS

############## Selection functions

## UCBSelection
# Upper confidence bound selection.
# @param current - the current tree node
# @param budget - the budget of the algorithm, if needed.
# @param data - generic data, if needed.
def UCBSelection(current, budget, data):
    bestScore = -np.inf
    bestChild = None

    for child in current.children:
        score = child.reward() + np.sqrt((2 * np.log(current.num_updates)) / child.num_updates)

        if score > bestScore:
            bestScore = score
            bestChild = child
    return bestChild

## UCBSelection
# Upper confidence bound selection for pareto fronts.
# Developed from a paper that I am too lazy to find cite to right now.
# @param current - the current tree node
# @param budget - the budget of the algorithm, if needed.
# @param data - generic data, if needed.
def paretoUCBSelection(current, budget, data):
    D = current.sum_reward.shape[0]
    #optimal = ParetoFront(D, len(current.children))

    UCB = [child.reward() +  \
            np.sqrt((4 * np.log(current.num_updates)) / (2 * child.num_updates)) \
            for child in current.children]
    # for child in current.children:
    #     UCB = child.reward() +  \
    #         np.sqrt((4 * np.log(current.num_updates)) / (2 * child.num_updates))
        #optimal.check_and_add(UCB, child)
    pareto_idx = get_pareto(np.array(UCB))
    rand_idx = np.random.randint(0, len(pareto_idx))

    child = current.children[pareto_idx[rand_idx]]
    #rew, child = optimal.get_random()
    return child


############## rollout functions

## randomRollout
# This function performs rollout using random child selection.
# @param treeState - the current state of the rollout function
# @param budget - the budget of the sequence
# @param data - a generic structure to store data for a rollout.
# @param keepEdges - [opt] if true, keep the edges in the path.
# @param keepNodes - [opt] if true, keep the nodes in the path.
#
# @return sequence of states. of roll out.
def randomRollout(treeState, budget, data=None, keepEdges=False, keepNodes=True):
    current = treeState
    remainingBudget = budget - treeState.rCost

    while True:
        succ = current.successor(budget)

        if len(succ) <= 0:
            break

        childIdx = np.random.randint(0, len(succ))
        child = succ[childIdx]

        current = child

    # return the path of the current state.
    return current.getPath(keepEdges=keepEdges, keepNodes=keepNodes)

    ## this is an older version that does not work with MCTSTree for
    # the sequence.
    # This should make the random rollout slightly slower, but the MCTS faster.
    # overall.
    '''current = treeState.state
    remainingBudget = budget - treeState.rCost

    while remainingBudget > 0:
        succ = current.successor()
        #pdb.set_trace()
        if len(succ) <= 0:
            break
        childIdx = np.random.randint(0, len(succ))
        child = succ[childIdx][0]
        edgeCost = succ[childIdx][1]

        #if remainingBudget > edgeCost:
        #sequence += [child]
        remainingBudget -= edgeCost
        current = child
        #else:
        #    break
    return current.getPath(keepEdges=keepEdges, keepNodes=keepNodes)
    '''

    '''succ = treeState.successor()
    if len(succ) == 0:
        return treeState

    childIdx = np.random.randint(0, len(succ))

    if succ[childIdx].cost() > budget:
        return treeState
    else:
        treeState.children.append(succ[childIdx])
        return randomRollout(succ[childIdx], budget, data)
    '''


############### solution functions



## bestAvgNext
# This function selects the next action which optimizes the best average reward.
# @param root - the current state of the rollout function
# @param bestSeq - the sequence with highest reward
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
# @param keepEdges - [opt] if true, keep the edges in the path.
# @param keepNodes - [opt] if true, keep the nodes in the path.
def bestAvgNext(root, bestSeq, bestR, data=None, keepEdges=False, keepNodes=True):
    best = -np.inf
    bestIdx = -1

    for i in range(len(root.children)):
        child = root.children[i]
        if child.reward() > best:
            best = child.reward()
            bestIdx = i

    if bestIdx != -1:
        return root.children[bestIdx].getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.children[bestIdx].reward()
    else:
        return root.getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.reward()

def mostSimulationsSingle(root, bestSeq, bestR, data=None, keepEdges=False, keepNodes=True):
    best = -np.inf
    bestIdx = -1
    for i in range(len(root.children)):
        child = root.children[i]
        if child.num_updates > best:
            best = child.num_updates
            bestIdx = i

    if bestIdx != -1:
        return root.children[bestIdx].getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.children[bestIdx].reward()
    else:
        return root.getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.reward()


def mostSimulations(root, bestSeq, bestR, data=None, keepEdges=False, keepNodes=True):
    # base case
    if len(root.children) < 1:
        return root.getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.reward()

    best = -np.inf
    bestIdx = -1
    for i in range(len(root.children)):
        child = root.children[i]
        if child.num_updates > best:
            best = child.num_updates
            bestIdx = i

    if bestIdx != -1:
        # recursively call mostSimulations on the best child
        return mostSimulations(root.children[bestIdx], bestSeq, bestR, data, keepEdges=keepEdges, keepNodes=keepNodes)
    else:
        return root.getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.reward()

## bestReward
# This function selects the best seen leaf
# @param root - the current state of the rollout function
# @param bestSeq - the sequence with highest reward
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
# @param keepEdges - [opt] if true, keep the edges in the path.
# @param keepNodes - [opt] if true, keep the nodes in the path.
def bestAvgReward(root, bestSeq, bestR, data=None, keepEdges=False, keepNodes=True):
    best = -np.inf
    bestIdx = -1

    for i in range(len(root.children)):
        child = root.children[i]
        if child.reward() > best:
            best = child.reward()
            bestIdx = i

    if bestIdx != -1:
        return bestAvgReward(root.children[bestIdx], bestSeq, bestR, data, keepEdges=keepEdges, keepNodes=keepNodes)
    else:
        return root.getPath(keepEdges=keepEdges, keepNodes=keepNodes), root.reward()

## highestReward
# This function selects the best seen leaf. (Should not be selected for multiple agents)
# The best sequence passed is the best reward returned regardless of the agent
# receiving that reward.
# @param root - the current state of the rollout function
# @param bestSeq - the sequence with highest reward
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
def highestReward(root, bestSeq, bestR, data=None):
    return bestSeq, bestR





#
