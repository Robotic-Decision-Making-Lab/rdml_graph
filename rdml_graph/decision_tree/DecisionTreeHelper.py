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
# DecisionTreeHelper.py
# Written Ian Rankin - April 2021
#
# A set of function to help the decision tree learner.

from rdml_graph.decision_tree import DecisionNode, FloatDecision, CategoryDecision

from scipy.stats import entropy
import numpy as np
import math
from itertools import combinations
from statistics import variance, mean

######################## IMPORTANCE FUNCTIONS #########################


# class_importance
# This is an entropy function on classification
def classification_importance(splits):
    # find all class numbers.
    class_nums_list = []

    total_entropy = 0
    total_num_samples = 0
    for i, X in enumerate(splits):
        # find number of class numbers for each split
        class_nums = {}
        for x in X:
            if isinstance(x[1], list) or isinstance(x[1], tuple):
                target = x[1][0]
            else:
                target = x[1]
            if target in class_nums:
                class_nums[target] += 1
            else:
                class_nums[target] = 1
        class_nums_list.append(class_nums)
        total_num_samples += len(X)

    # find entropy of splits
    for i, class_nums in enumerate(class_nums_list):
        cur_entropy = entropy(np.array(list(class_nums.values())))
        total_entropy += cur_entropy * (len(splits[i]) / total_num_samples)

    return -total_entropy


def regression_importance(splits):
    loss = 0

    total_num_samples = 0
    variances = []

    for X in splits:
        targets = [x[1] for x in X]
        if len(targets) > 0:
            if isinstance(targets[0], list) or isinstance(targets[0], tuple):
                targets_raw = targets
                targets = [t[0] for t in targets]

        if len(targets) <= 1:
            variances.append(0)
        elif len(targets) == 2:
            mean = targets[0] + targets[1]
            sum_var = (targets[0]-mean)**2+(targets[1]-mean)**2
            variances.append(sum_var/2)
        else:
            variances.append(variance(targets))

        total_num_samples += len(X)

    output_var = 0
    for i in range(len(splits)):
        output_var += variances[i] * (len(splits[i]) / total_num_samples)

    return -output_var



####################### PLURALITY FUNCTIONS ########################


def class_plurality(X):
    if len(X) == 0:
        raise ValueError("class_plurality passed an empty list")

    class_nums = {}
    most_class = None
    most_number = 0

    for x, y in X:
        if y not in class_nums:
            num = 1
        else:
            num = class_nums[y] + 1

        class_nums[y] = num
        if num > most_number:
            most_number = num
            most_class = y

    return most_class

def reg_plurality(X):
    if len(X) == 0:
        raise ValueError("reg_plurality passed an empty list")
    if len(X) == 1:
        return X[0][1]
    else:
        if isinstance(X[0][1], list) or isinstance(X[0][1], tuple):
            targets = [x[1][0] for x in X]
            return [x[1] for x in X]
        else:
            targets = [x[1] for x in X]

        return mean(targets)
    #else:



############################### STOP FUNCTIONS ######################

# check if the same class
# @param X - the input data [[sample, class], ...]
#
# @return - true if all data is the same class, otherwise false
def same_class(X):
    if len(X) == 0:
        return True
    else:
        first = X[0][1]

        # check if all of the same class.
        for i in range(1, len(X)):
            if X[i][1] != first:
                return False
        return True


############################ SPLIT FUNCTIONS ###########################


# bin_category_split
# A binary split of categories using the importance function to determine the
# best split.
# @param X - the input data [(x_i, target), ...]
# @param importance_func - the importance function for the split
def bin_category_split(X, importance_func, parent, id):
    categories = set()

    for (v, target) in X:
        if v not in categories:
            categories.add(v)

    num_categories = len(categories)

    if num_categories == 1:
        return None, -float('inf')

    best_atr = None
    best_importance = -float('inf')

    for r in range(1, int(math.ceil((num_categories+1)/2))):
        comb = combinations(categories, r)

        # go through each combination of splits
        for atr in comb:
            atr = set(atr)
            # find inverse set
            #atr_not = categories - atr

            splits = [[],[]]
            for x in X:
                if x[0] in atr:
                    splits[0].append(x)
                else:
                    splits[1].append(x)
            # end for x in X

            # find importance value
            importance = importance_func(splits)
            if importance > best_importance:
                best_importance = importance
                best_atr = atr
        # for atr in comb
    # end for r in range
    cats = [list(best_atr), list(categories - best_atr)]
    n = CategoryDecision(id, parent, -1, cats)
    return n, best_importance


def data_sort_key(x):
    return x[0]

# bin_float_split
# A binary split of floats using the importance function to determine the
# best split.
# This function sorts by the given data and incrementally finds the best split
# @param X - the input data [(x_i, target), ...]
# @param importance_func - the importance function for the split
def bin_float_split(X, importance_func, parent, id, with_labels=True):
    X.sort(key=data_sort_key)

    best_atr = None
    best_importance = -float('inf')

    if with_labels:
        x_prev = X[0][0]
    else:
        x_prev = X[0]

    for i in range(1, len(X)):
        if with_labels:
            x_i = X[i][0]
        else:
            x_i = X[i]

        if x_prev == x_i:
            # if they are the same value, don't bother splitting here
            continue

        splits = [X[0:i], X[i:]]
        importance = importance_func(splits)

        # update the best importance if needed
        if importance > best_importance:
            best_importance = importance
            best_atr = i

    if best_atr is None:
        # There are no viable splits
        return None, best_importance

    if with_labels:
        value = (X[best_atr-1][0] + X[best_atr][0]) / 2
    else:
        value = (X[best_atr-1] + X[best_atr]) / 2
    # end for loop for all samples
    n = FloatDecision(id, parent, -1, value)
    return n, best_importance

######################### ATTRIBUTE FUNCTIONS ##########################
# These attribute functions can be modified to particular types of objects.
# Also modded using importance functions and splitter functions.



# default_attribute_func
# Finds the argmax of the possible attributes given a particular importance_func
# @param X - the input data [(x1, label1), ...]
# @param importance_func - the importance of different attributes
# @param types - list of types for x vector ['float', 'category']
# @param parent - the parent node for this attribute function
#
# @return - the best node given these possible types.
def default_attribute_func(X, importance_func, types, parent, id):
    if len(X) <= 0:
        return None

    attribute_handlers = {'float': bin_float_split, 'category': bin_category_split}

    best_attribute = None
    best_importance = -float('inf')
    #best_split = None

    num_dim_x = len(X[0][0])

    for i in range(num_dim_x):
        handler = attribute_handlers[types[i]]
        X_i = [(x[0][i], x[1]) for x in X]
        #targets = [x[1] for x in X]

        n, importance = handler(X_i, importance_func, parent, id)
        # set index of node to correct index
        if n is not None:
            n.idx = i

        if importance > best_importance:
            best_importance = importance
            best_attribute = n
    return best_attribute
