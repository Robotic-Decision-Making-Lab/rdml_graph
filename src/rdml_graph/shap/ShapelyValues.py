# ShapelyValues.py
# Written Ian Rankin - June 2021
#
# A set of methods to calculate Shapely values.
#
# References:
# [1] From Local Explanation to Global Understanidng with Explainable AI for Trees
#       (2020) Scott Lundberg, et al.

import rdml_graph as gr
from rdml_graph.decision_tree import DecisionNode, Ensemble


import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

import numpy as np
import math
from statistics import mean

from multiprocessing import Pool

import pdb


## this calculates the SHAP values for the input feature x
# These are approximate SHAP Values.
# Likely better to instead use TreeSHAP_INT, which uses interventional samples
# from the decision tree to better approximate the SHAP values.
# @param x - the input feature
# @param tree - the input tree
#
# @return - list of SHAP values
def TreeSHAP(x, tree):
    return [TreeSHAP_idx(x, i, tree) for i in range(len(x))]

# TreeSHAP_idx
# This returns the approximate SHAP value for a single feature index
# Algorithm 1 from [1]
# @param x - the feature input to the tree
# @param feature_idx - the index to check on the feature
# @param tree - the input node to caluclate the index from
#
# @return - the SHAP value for feature x for index feature_idx
def TreeSHAP_idx(x, feature_idx, tree):
    print(tree)



    # base case - check if tree is a leaf node
    if not isinstance(tree, gr.DecisionNode):
        # TODO make sure this is valid
        val = leaf_value(tree)
        print('tree: ' + str(tree) + ' val = ' + str(val))
        #pdb.set_trace()
        return val


    # check if we are conditioning on this feature.
    if tree.idx == feature_idx:
        # use child on the decision path
        #pdb.set_trace()

        return TreeSHAP_idx(x, feature_idx, tree.get_next(x))
    else:
        # weight children by their coverage (number of samples)
        sum_var = 0
        #pdb.set_trace()
        splits = tree.separate(tree.samples)

        for i, e in enumerate(tree.e):
            val = TreeSHAP_idx(x, feature_idx, e.c)
            #print('val: ' + str(val) + ' tree: ' + str(tree))
            if isinstance(val, tuple):
                print('tuple')
                pdb.set_trace()
            elif isinstance(splits[i], int):
                print('int')
                pdb.set_trace()

            sum_var += val * len(splits[i])

        return sum_var / len(tree.samples)


# get the value of the leaf node
# for multiple inputs, the mean of the target values
def leaf_value(leaf):
    if isinstance(leaf, list):
        if len(leaf) > 0:
            if isinstance(leaf[0], Sequence) or isinstance(leaf[0], tuple):
                if isinstance(leaf[0][0], (Sequence, np.ndarray)):
                    vals = [x[0][0] for x in leaf]
                else:
                    vals = [x[0] for x in leaf]
                # mean of the leaf objects
                return mean(vals)
            else:
                # no attached object just numbers
                return mean(leaf)
        else:
            print('leaf_value does not make sense, returning none')
            return None
    elif isinstance(leaf, tuple):
        if isinstance(leaf[0], Sequence):
            return leaf[0][0]
        else:
            return leaf[0]
    else:
        # not a list just return the raw value.
        return leaf




################################### TreeSHAP_interventional feature pert

## SHAP_avg_diff takes the average between the x sample and all samples in the
# the tree.
# @param x - the input feature vector to test (k,)
# @param tree - the input decision tree to calculate SHAP values from.
#
# @return diff_of_average (numpy (k,)), diff_of_median, all shap values (numpy (n,k))
def SHAP_avg_diff(x, tree):
    shap_values = SHAP_all(tree)
    feat_shap = np.array(TreeSHAP_INT(x, tree))

    avg_shap = np.mean(shap_values, axis=0)
    med_shap = np.median(shap_values, axis=0)

    print('AVG SHAP: ' + str(avg_shap))
    print('MED SHAP: ' + str(med_shap))

    return feat_shap - avg_shap, feat_shap - med_shap, shap_values


def calc_tree_shap_int_all_for_parallel(tree, samples):
    return np.array([TreeSHAP_INT(s, tree) for s in samples])

## SHAP_val
# This function takes as input the input decision tree and finds the SHAP value
# for all samples in the tree
# @param tree - the input decision tree to run the function over
#
# @return - numpy array of shap values (n, k) (number of samples, number of features)
def SHAP_all(tree, num_threads=8):
    print('SHAP_all called')
    samples = [c[0] for c in tree.samples]
    print('Generated samples')

    if isinstance(tree, Ensemble):
        print('Determined tree is Ensemble')
        iteratable = [(t, samples) for t in tree.trees]
        print('Created iteratable')

        with Pool(num_threads) as p:
            print('With pool: ')
            print(p)
            all_shaps = p.starmap(calc_tree_shap_int_all_for_parallel, iteratable)

        #pdb.set_trace()
        all_shaps = np.array(all_shaps)

        shap_values = np.average(all_shaps, axis=0, weights=tree.weights)

    else:
        shap_values = np.array([TreeSHAP_INT(s, tree) for s in samples])

    return shap_values



## TreeShap with interventional feature pertubation
# This implements algorithm TBD from [1]
# @param x - the input feature
# @param tree - the input tree
#
# @return - the shap value for the given input feature.
def TreeSHAP_INT(x, tree):
    phi = np.zeros(len(x))

    refset = [c[0] for c in tree.samples]

    for c in refset:
        xlist = np.zeros(len(x))
        clist = np.zeros(len(x))
        SHAP_recurse(tree, 0, 0, xlist, clist, x, c, phi)

    return phi / len(refset)

# Shapely value weight for a set size and number of features.
def calc_weight(U, V):
    return math.factorial(U) * math.factorial(V-U-1) / math.factorial(V)

## TODO multiply by Vj (tree)???

def SHAP_recurse(tree, U, V, xlist, clist, x, c, phi):
    # base case - check if tree is a leaf node
    if not isinstance(tree, gr.DecisionNode):
        #pdb.set_trace()
        pos = neg = 0
        if U == 0:
            return (pos, neg)
        leaf_val = leaf_value(tree)
        if U != 0:
            pos = calc_weight(U-1, V) * leaf_val
        if U != V:
            neg = -calc_weight(U,V) * leaf_val
        return (pos, neg)

    next_tree = None
    x_next = tree.get_next(x)
    c_next = tree.get_next(c)
    if x_next == c_next:
        next_tree = x_next

    if xlist[tree.idx] > 0:
        next_tree = x_next
    if clist[tree.idx] > 0:
        next_tree = c_next

    if next_tree is not None:
        return SHAP_recurse(next_tree, U, V, xlist, clist, x, c, phi)
    else:
        # recurse left and right
        xlist[tree.idx] += 1
        posx, negx = SHAP_recurse(x_next, U+1, V+1, xlist, clist, x, c, phi)
        xlist[tree.idx] -= 1
        clist[tree.idx] += 1
        posc, negc = SHAP_recurse(c_next, U, V+1, xlist, clist, x, c, phi)
        clist[tree.idx] -= 1

        phi[tree.idx] += posx + negc
        #print('phi: ' + str(phi))
        return posx + posc, negx + negc



## Select the shapely index with interesting features.
# Only keep the indicies that have a certian percentage over (forced to select one)
# This function selects the indicies of features with different shap values
# @param shap - the vector of the given shap value
# @param shap_diff - the difference of shap values for the given alt
# @param max_select - the max number of features to select
# @param min_select - the minumum number of features to select
# @param
def select_SHAP_dynamic(shap, shap_diff, max_select, min_select=1, \
                        perc_to_select=0.05, isMax=True):
    sort_idx = np.argsort(shap_diff)
    if not isMax:
        sort_idx = sort_idx[::-1]
        shap_diff_cur = -shap_diff

    largest = len(sort_idx)-1
    selected = []
    avg_shap = np.mean(shap)
    if avg_shap == 0:
        avg_shap = 1

    while len(selected) < max_select and largets >= 0:
        # check the value is within the percentage to select, if not reject.
        # force to keep the min to select values.
        if len(selected) > min_select and \
                (shap_diff_cur[sort_idx[largest]] / avg_shap) < perc_to_select:
            break

        selected.append(sort_idx[largest])
        largest -= 1

    return selected


## select the shapely index by keeping the largest values first. If it is out of
# positive values it moves to the smallest negative value. 0 values are ignored.
# @param SHAP_diff - the difference of the average/median SHAP value
# @param k_to_select - the number of indicies to select and return
#
# @return indicies to select []
def select_SHAP_idx(SHAP_diff, k_to_select, isMax=True):
    if k_to_select < 0:
        raise ValueError('select_SHAP_idx cannot handle less indicies then selected')

    sort_idx = np.argsort(SHAP_diff)
    if not isMax:
        sort_idx = sort_idx[::-1]
        SHAP_diff = -SHAP_diff

    largest = len(sort_idx)-1
    smallest = 0
    selected = []

    while len(selected) < k_to_select and largest >= smallest and \
            largest >= 0 and smallest < len(sort_idx):
        if SHAP_diff[sort_idx[largest]] > 0:
            selected.append(sort_idx[largest])
            largest -= 1
        elif SHAP_diff[sort_idx[smallest]] < 0:
            selected.append(sort_idx[smallest])
            smallest += 1
        else:
            break

    return selected











#################################### Test functions #######################




def main():
    test_list = [0.01]
    print(select_SHAP_idx(test_list, 3))

    num_samps = 100
    num_dim = 2

    xs = [[random.uniform(0,10) for j in range(num_dim)]  for i in range(num_samps)]
    X = [(x, f(x)+random.uniform(-0.5, 0.5)) for x in xs]

    #print(xs)
    #print(X)

    types = ['float'] * num_dim

    root,_ = gr.learn_decision_tree(X, \
                    types=types, \
                    attribute_func=gr.default_attribute_func,\
                    importance_func=gr.regression_importance, \
                    plurality_func=gr.reg_plurality,\
                    max_depth=100)

    t = root.get_viz(labels=True)
    t.view()

    x = [5,5]

    prediction = root.traverse(x)
    shap = TreeSHAP(x, root)

    print('TreeSHAP_int')
    shap_int = TreeSHAP_INT(x, root)

    print(prediction)
    print(shap)
    print(shap_int)
    print(sum(shap))


if __name__ == '__main__':
    import sklearn.datasets as dt
    import matplotlib.pyplot as plt
    import random

    def f(x_in):
        y_0 = np.sin(x_in[0]-.3572)*4-0.2
        y_1 = (np.cos(x_in[1]*1.43)-.3572)*3
        #y_2 = x_in[2]

        return y_0 + y_1# + y_2

    main()
