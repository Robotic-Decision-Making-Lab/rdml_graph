# MCTSTree.py
# Written Ian Rankin February 2020
# Based on code written by Graeme Best, and also code written by Seth McCammon
#
# A state of the MCTS tree which contains various helper functions for
# the tree search.

from rdml_graph import State
from rdml_graph import SearchState

class MCTSTree(SearchState):
    # Constructor

    def __init__(self, state, rCost, parent, ):
        super(MCTSTree, self).__init__(state, rCost=rCost, hCost=0 parent=parent)

        self.unpicked_children = 
        self.num_updates = 0
