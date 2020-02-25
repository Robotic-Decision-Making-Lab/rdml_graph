# MCTSHelper.py
# Written Ian Rankin February 2020
#
# A set of various helper functions for the MCTS Tree search.

import tqdm
import numpy as np
from rdml_graph import MCTSTree

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
def randomRollout(treeState, budget, data=None):
    succ = treeState.successor()
    if len(succ) == 0:
        return treeState

    childIdx = np.random.randint(0, len(succ))

    if succ[childIdx].cost() > budget:
        return treeState
    else:
        treeState.children.append(succ[childIdx])
        return randomRollout(succ[childIdx], budget, data)


############### solution functions

# bestReward
# This function performs rollout using random child selection.
# @param root - the current state of the rollout function
# @param bestLeaf - the best possible leaf
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
def bestReward(root, bestLeaf, bestR, data=None):
    return bestLeaf
