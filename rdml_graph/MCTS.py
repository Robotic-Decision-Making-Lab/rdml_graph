# MCTS.py
# Written Ian Rankin February 2020
# Based on code written by Graeme Best, and also code written by Seth McCammon
#
# An implementation of the monte-carlo tree search algorithm
#


import tqdm
import numpy as np
from rdml_graph import MCTSTree
from rdml_graph import UCBSelection, randomRollout, bestReward

import pdb


# MCTS
# The main entry function to the MCTS algorithm.
# @param start - the entry state of the MCTS algorithm
# @param max_iterations - maximum number of iterations the MCTS algorithm runs.
# @oaram budget - the total budget of
def MCTS(   start, max_iterations, rewardFunc, budget=1.0, selection=UCBSelection, \
            rolloutFunc=randomRollout, solutionFunc=bestReward, data=None, actor_number=0):
    # Set the root of the search tree.
    root = MCTSTree.MCTSTree(start, 0, None)
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
                if len(current.children) <= 0:
                    # reached planning horizon, perform rollout on this node.
                    break

                current = selection(current, budget, data)
        # end selection expansion while loop.

        ######## ROLLOUT
        # perform rollout to the end of a possible sequence.
        leaf = rolloutFunc(current, budget, data)
        rolloutReward, rewardActorNum = rewardFunc(leaf.getPath(), budget, data)

        # Keep track of best reward
        if rolloutReward > bestReward and actor_number == rewardActorNum:
            bestReward = rolloutReward
            bestLeaf = leaf

        ######## BACK-PROPOGATE
        leaf.backpropReward(rolloutReward, rewardActorNum)

    # end main for loop
    pdb.set_trace()

    ######## SOLUTION
    solutionLeaf = solutionFunc(root, bestLeaf, bestReward, data)
    return solutionLeaf.getPath()
# End MCTS
