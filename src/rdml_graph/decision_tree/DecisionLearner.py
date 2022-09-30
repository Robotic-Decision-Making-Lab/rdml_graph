# Copyright 2021 Ian Rankin
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
## @package DecisionLearner.py
# Written Ian Rankin - April 2021
#
# Set of functions to learn decision tree's

from rdml_graph.core import TreeNode
from rdml_graph.decision_tree.DecisionTreeHelper import \
        classification_importance, regression_importance, \
        class_plurality, reg_plurality, \
        same_class, bin_category_split, bin_float_split, default_attribute_func

import pdb



## learn_decision_tree from the data in parameter X
# @param X - the input data
# REQUIRED
# @param importance_func - the importance of different attributes
# @param attribute_func - the function to find and generate the next best node
#       of the decision tree must be
#       type - func(X, importance_func, types, parent)
#
# OPTIONAL
# @param plurality_func - [opt] return the leaf node for a particular part of the tree
# @param same_func - checks if the data input is all the same and spliting
#               can stop.
# @param types - [opt] a list of types of the input tree to use with the attribute function
# @param with_labels - [opt] defines if it is a classification or regression problem with labels on X
# @param max_depth - [opt] the max_depth to allow the tree to go
#
# FOR RECURSION (DON'T USE)
# @param cur_depth - [DO NOT USE] sets the current depth of the tree
# @param parent - [DO NOT USE] the parent of the node to learn.
# @param parent_samples - [DO NOT USE] the samples of the parent node
#
# RETURN
# @return the root of the sub-tree of decision nodes OR the leaf of the tree
def learn_decision_tree(X, \
        types, \
        importance_func, \
        attribute_func= default_attribute_func, \
        max_depth=float('inf'),\
        plurality_func=class_plurality, \
        same_func=same_class, \
        with_labels=True, \
        cur_depth=0, \
        parent=None, \
        parent_samples=None,
        id = 0):
    # Start function
    #pdb.set_trace()

    # Check for base case
    if len(X) == 0:
        if parent_samples is None:
            raise ValueError('Root of tree passed no samples')
        return plurality_func(parent_samples), id
    if cur_depth >= max_depth or same_func(X):
        # base class
        return plurality_func(X), id
    else:
        # find best case
        n = attribute_func(X, importance_func, types, parent, id, with_label=with_labels)
        if n is None:
            #print('Attribute function returned None, and I do not know why, debug this!')
            #pdb.set_trace()
            #print('\n\n\nX: ' + str(X))
            return plurality_func(X), id
        n.samples = X
        n.types = types
        splits = n.separate(X, with_label=with_labels)

        for i, sub_X in enumerate(splits):
            id += 1
            child, id = learn_decision_tree( \
                    X=sub_X, \
                    importance_func=importance_func, \
                    attribute_func=attribute_func, \
                    max_depth=max_depth, \
                    plurality_func=plurality_func, \
                    same_func=same_func, \
                    types=types, \
                    with_labels = with_labels, \
                    cur_depth = cur_depth + 1, \
                    parent = n, \
                    parent_samples = X, \
                    id = id)

            n.set_node(i, child)
        # return sub-tree or full tree.
        return n, id
    # end if else for base case or non-base case
