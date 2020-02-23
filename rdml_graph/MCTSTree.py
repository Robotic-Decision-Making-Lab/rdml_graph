# MCTSTree.py
# Written Ian Rankin February 2020
# Based on code written by Graeme Best, and also code written by Seth McCammon
#
# A state of the MCTS tree which contains various helper functions for
# the tree search.

from rdml_graph import State
from rdml_graph import SearchState

import numpy as np

class MCTSTree(SearchState):
    # Constructor

    def __init__(self, state, rCost, parent):
        super(MCTSTree, self).__init__(state, rCost=rCost, hCost=0, parent=parent)

        self.unpicked_children = []
        self.children = []
        self.sum_reward = 0
        self.num_updates = 0

    # reward
    # reward is an average of all total rewards propagated through the tree.
    # @return - reward estimate of current state.
    def reward(self):
        return self.sum_reward / self.num_updates

    def backpropReward(self, reward):
        self.sum_reward  += reward
        self.num_updates += 1

        if self.parent is not None:
            self.parent.backpropReward(reward)

    # expandNode
    # expands the current node at the given index.
    # If idx is none, then a random unpicked child is selected.
    # @param idx - the index of the unpicked child to expand (normaly random)
    #
    # @return - the MCTSTree to expand
    # @post - unpicked children has child removed, added to children
    def expandNode(self, idx = None):
        childIdx = np.random.randint(0, len(self.unpicked_children))
        child = self.unpicked_children[childIdx]
        del self.unpicked_children[childIdx]
        self.children.append(child)

        return child


    # successor
    # successor function for states that removes children over budget.
    # @param budget - the max budget of the successor function.
    #
    # @return - list of MCTSTree objects.
    def successor(self, budget=np.inf):
        result = []
        for s in self.state.succesor():
            cost = self.rCost + s[1]
            if cost <= budget:
                result.append(MCTSTree(s[0], cost, self))
        return result
