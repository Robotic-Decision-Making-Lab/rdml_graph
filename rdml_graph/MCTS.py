# MCTS.py
# Written Ian Rankin February 2020
# Based on code written by Graeme Best, and also code written by Seth McCammon
#
# An implementation of the monte-carlo tree search algorithm
#


import tqdm
import numpy as np
from rdml_graph import MCTSTree



# MCTS
# The main entry function to the MCTS algorithm.
# @param start - the entry state of the MCTS algorithm
# @param max_iterations - maximum number of iterations the MCTS algorithm runs.
# @oaram budget - the total budget of
def MCTS(   start, max_iterations, rewardFunc, budget=1.0, selection=UCBSelection, \
            rolloutFunc=randomRollout, solutionFunc=bestReward, data=None):
    # Set the root of the search tree.
    root = MCTSTree(start, 0, None)
    root.unpicked_children = root.successor()

    bestReward = -np.inf
    bestLeaf = None

    # Main loop of MCTS
    for i in tqdm.tqdm(range(max_iterations)):
        current = root

        ######### SELECTION and Expansion
        # Check all possibilties of selection.
        while True:
            if len(current.unpicked_children) > 0:
                ######## Expansion
                child = current.expandNode(budget)
                child.unpicked_children = child.successor()

                current = child

                # once a node has been successfully expanded break out of selection loop.
                break
            else:
                ######## Selection.
                current = selection(current, budget, data)

                break
        # end selection expansion while loop.

        ######## ROLLOUT
        # perform rollout to the end of a possible sequence.
        leaf = rolloutFunc(current, budget, data)
        rolloutReward = rewardFunc(leaf.getPath(), budget, data)

        # Keep track of best reward
        if rolloutReward > bestReward

        ######## BACK-PROPOGATE
        leaf.backpropReward()

    # end main for loop

    ######## SOLUTION
    solutionLeaf = solutionFunc(root, bestLeaf, bestReward, data)
    return solutionLeaf.getPath()
# End MCTS

########################### Common functions for MCTS

############## rollout functions

# randomRollout
# This function performs rollout using random child selection.
# @param treeState - the current state of the rollout function
# @param budget - the budget of the sequence
# @param data - a generic structure to store data for a rollout.
def randomRollout(treeState, budget, data=None):
    succ = treeState.successor()
    if len(succ) == 0:
        return self

    childIdx = np.random.randint(0, len(succ))

    if succ[childIdx].cost() > budget:
        return self
    else:
        self.children.append(succ[childIdx])
        return randomRollout(succ[childIdx])

############### solution functions

# bestReward
# This function performs rollout using random child selection.
# @param root - the current state of the rollout function
# @param bestLeaf - the best possible leaf
# @param bestR - the best seen reward
# @param data - generic data possibly useful for the best reward.
def bestReward(root, bestLeaf, bestR, data=None):
    return bestLeaf
