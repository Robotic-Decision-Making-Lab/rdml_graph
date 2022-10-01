# BalancedTree.py
# Written Ian Rankin - September 2022, based on previous code I wrote in 2021.
#
# This creates a balanced decision tree, useful as a kd-tree or other balanced set of trees.
#

from rdml_graph.decision_tree import learn_decision_tree



from rdml_graph.decision_tree.DecisionTreeHelper import reg_plurality, balance_float_split

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
            with_labels=False)


    return n


def input_plurality(X):
    return X


## balanced_importance
# counts the number of splits
def balanced_importance(splits):
    diff_num = abs(len(splits[0]) - len(splits[1]))
    
    return -diff_num










