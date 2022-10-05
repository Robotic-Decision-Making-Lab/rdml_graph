# BalancedTree.py
# Written Ian Rankin - September 2022, based on previous code I wrote in 2021.
#
# This creates a balanced decision tree, useful as a kd-tree or other balanced set of trees.
#

from rdml_graph.decision_tree import learn_decision_tree
from rdml_graph.decision_tree.DecisionTreeHelper import reg_plurality, balance_float_split

import pdb

## create_balanced_decision_tree
# @param X - the input samples [list of n x m samples must be floats]
#
# @return 
def create_balanced_decision_tree(X):

    types = ['float_balance'] * len(X[0])

    # learn the decision tree
    n = learn_decision_tree(X,
            types = types,
            importance_func=balanced_importance,
            plurality_func=input_plurality, \
            same_func = never_same, \
            with_labels=False, \
            X_only=True)


    return n


def input_plurality(X):
    return X[0]

def never_same(X):
    #print(len(X))
    if len(X) <= 1:
        return True
    else:
        in0 = X[0]
        #print(X)
        for x in X[1:]:
            if (x != in0).any():
                return False 
        return True

## balanced_importance
# counts the number of splits
def balanced_importance(splits):
    diff_num = abs(len(splits[0]) - len(splits[1]))
    
    return -diff_num










