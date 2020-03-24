# MCTSHelper.py
# Written Ian Rankin February 2020
#
# A set of various helper functions for the MCTS Tree search.

import tqdm
import numpy as np
from . import MCTSTree

import pdb

########################### Common functions for MCTS

############## Selection functions

# UCBSelection
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
############## rollout functions

# randomRollout
# This function performs rollout using random child selection.
# @param treeState - the current state of the rollout function
# @param budget - the budget of the sequence
# @param data - a generic structure to store data for a rollout.
#
# @return sequence of states. of roll out.
def randomRollout(treeState, budget, data=None):
    sequence = treeState.getPath()

    current = treeState.state
    remainingBudget = budget - treeState.rCost

    while remainingBudget > 0:
        succ = current.successor()
        #pdb.set_trace()
        if len(succ) <= 0:
            break
        childIdx = np.random.randint(0, len(succ))
        child = succ[childIdx][0]
        edgeCost = succ[childIdx][1]

        if remainingBudget > edgeCost:
            sequence += [child]
            remainingBudget -= edgeCost
            current = child
        else:
            break
    return sequence


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



# bestAvgNext
# This function selects the next action which optimizes the best average reward.
# @param root - the current state of the rollout function
# @param bestSeq - the sequence with highest reward
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
def bestAvgNext(root, bestSeq, bestR, data=None):
    best = -np.inf
    bestIdx = -1

    for i in range(len(root.children)):
        child = root.children[i]
        if child.reward() > best:
            best = child.reward()
            bestIdx = i

    if bestIdx != -1:
        return root.children[bestIdx].getPath(), root.children[bestIdx].reward()
    else:
        return root.getPath(), root.reward()

def mostSimulations(root, bestSeq, bestR, data=None):
    best = -np.inf
    bestIdx = -1
    for i in range(len(root.children)):
        child = root.children[i]
        if child.num_updates > best:
            best = child.num_updates
            bestIdx = i

    if bestIdx != -1:
        return root.children[bestIdx].getPath(), root.children[bestIdx].reward()
    else:
        return root.getPath(), root.reward()

# bestReward
# This function selects the best seen leaf
# @param root - the current state of the rollout function
# @param bestSeq - the sequence with highest reward
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
def bestAvgReward(root, bestSeq, bestR, data=None):
    best = -np.inf
    bestIdx = -1

    for i in range(len(root.children)):
        child = root.children[i]
        if child.reward() > best:
            best = child.reward()
            bestIdx = i

    if bestIdx != -1:
        return bestAvgReward(root.children[bestIdx], bestSeq, bestR, data)
    else:
        return root.getPath(), root.reward()

# highestReward
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
