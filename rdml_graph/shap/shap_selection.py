# shap_selection.py
# Written Ian Rankin August 2021
#
# a set of functions for picking paths using the shap values
# This is opposed to selecting explanations via the decision tree.

import numpy as np
import sys
from rdml_graph.shap import select_SHAP_idx


## select_alts_from_shap_diff
# @param best_path_idx - the index of the best path selected
# @param shap_values - list of shapely values for each path features from the best path (numpy nxm)
# @param num_paths - the number of paths to show. (includes best path)
# @param selection_method - string ['random_less', 'worse_shap', 'similar_except']
#
# @return alts_to_show (list), pertinent_features (list)
def select_alts_from_shap_diff(best_path_idx, shap_values, num_paths, selection_method='random_less', isMax=True, exclude_func=None, data=None):
    best_path_shap = shap_values[best_path_idx]
    median_shap = np.median(shap_values, axis=0)

    shap_diff = best_path_shap - median_shap
    # Select the most pertinent features
    pertinent_features = select_SHAP_idx(shap_diff, num_paths-1, isMax=isMax)

    # Select the best alternatives to show for each pertinent feature.
    alts_to_show = [best_path_idx]
    for pert_feat in pertinent_features:
        alt_idx = -1
        greater = shap_diff[pert_feat] < 0
        # if not isMax:
        #     greater = not greater


        if selection_method == 'random_less':
            alt_idx = select_random_less(pert_feat, best_path_shap[pert_feat], \
                                        shap_values, greater, set(alts_to_show))
        elif selection_method == 'worse_shap':
            alt_idx = select_worse_shap(pert_feat, \
                                        shap_values, greater, set(alts_to_show))
        elif selection_method == 'similar_except':
            alt_idx = select_similar_except(best_path_idx, pert_feat, shap_values, \
                                greater, set(alts_to_show), exclude_func=exclude_func, data=data)

        alts_to_show += [alt_idx]

    return alts_to_show, pertinent_features


######################## Selection methods

## select_random_less
# This function selects shap values that are less then the given value at random
# @param shap_values - the list of shap values for the paths
# @param shap_idx - the index of the shap vector that is being evaluated
# @param value - the value of the shap vector to return with less than
# @param greater - [opt] set to false to invert and return indicies which are greater than
# @param exclude - [opt] set of indicies to exclude
def select_random_less(shap_idx, value, shap_values, greater=False, exclude_idx={}):
    # Get all alts with shap value less than the given selection.
    possible_alts = get_alts_with_feat_less_or_great(shap_values, shap_idx, value, greater, exclude_idx)

    if len(possible_alts) < 1:
        return -1

    rand_idx = np.random.randint(0, len(possible_alts))
    return possible_alts[rand_idx]


## select_worse_shap
# select the worst shap value on the given index
# @param shap_idx - the index of the shap vector that is being evaluated
# @param shap_values - the list of shap values for the paths
# @param greater - [opt] set to false to invert and return indicies which are greater than
# @param exclude - [opt] set of indicies to exclude
def select_worse_shap(shap_idx, shap_values, greater=False, exclude_idx={}):
    worse_idx = -1
    worse_val = float('inf')
    if greater:
        worse_val = -float('inf')
    else:
        worse_val = float('inf')

    for i in range(shap_values.shape[0]):
        if i not in exclude_idx:
            if greater:
                if shap_values[i,shap_idx] > worse_val:
                    worse_idx = i
                    worse_val = shap_values[i,shap_idx]
            else:
                if shap_values[i,shap_idx] < worse_val:
                    worse_idx = i
                    worse_val = shap_values[i,shap_idx]

    return worse_idx

## select_similar_except
# This tries to find the vector that is most similar to given selected index except
# along the given index. This does anr argmax of the sum of the L-norm and the difference in values
# @param sel_alt_idx - the selected alt index
# @param shap_values - the list of shap values for the paths
# @param shap_idx - the index of the shap vector that is being evaluated
# @param value - the value of the shap vector to return with less than
# @param greater - [opt] set to false to invert and return indicies which are greater than
# @param exclude - [opt] set of indicies to exclude
# @param order - the order of the norm on the second feature (typically a 2 norm)
def select_similar_except(sel_alt_idx, shap_idx, shap_values, greater=False, exclude_idx={}, order=2, exclude_func=None, data=None):
    selected_shap = shap_values[sel_alt_idx]
    scores = np.ones((shap_values.shape[0], 2))

    for idx in range(shap_values.shape[0]):
        if idx not in exclude_idx:
            shap_diff = selected_shap - shap_values[idx]
            feat_diff = shap_diff[shap_idx]
            if greater:
                feat_diff = -feat_diff # reverse direction for features that are greater than.

            if feat_diff < 0.000001:
                scores[idx, :] = -float('inf')
            if exclude_func is not None:
                if exclude_func(shap_values[idx], selected_shap, data):
                    scores[idx, :] = -float('inf')
            else:
                # find and set scores
                scores[idx,0] = feat_diff
                all_else_vec = shap_diff[np.arange(len(shap_diff))!=shap_idx]

                scores[idx,1] = -np.linalg.norm(all_else_vec, ord=order)
        else:
            scores[idx, :] = -float('inf')


    # with no particular math for combining...
    if selected_shap.shape[0] > 1:
        utility = scores[:,0] + scores[:,1]/((selected_shap.shape[0]-1)**(1/order))
    else:
        utility = scores[:,0]

    return np.argmax(utility)




############################ Helper functions

## Return indicies of alternatives with a shap_value less than or greater than
# the given value
# @param shap_values - the list of shap values for the paths
# @param shap_idx - the index of the shap vector that is being evaluated
# @param value - the value of the shap vector to return with less than
# @param greater - [opt] set to false to invert and return indicies which are greater than
# @param exclude - [opt] set of indicies to exclude
#
# @return list of indicies with a shap value[shap_idx] less than the given value.
def get_alts_with_feat_less_or_great(shap_values, shap_idx, value, greater=False, exclude_idx={}):
    indicies = []

    for idx, shap_value in enumerate(shap_values):
        if idx not in exclude_idx:
            if greater:
                if shap_value[shap_idx] > value:
                    indicies += [idx]
            else:
                if shap_value[shap_idx] < value:
                    indicies += [idx]

    return indicies





if __name__ == '__main__':
    # test code
    shap_values = np.random.random((16, 5))

    alts_to_show, features = select_alts_from_shap_diff(0, shap_values, 3, 'similar_except')
    print('Alts to show: ' + str(alts_to_show)+' pertinent_features: '+str(features))
    print(shap_values[alts_to_show[0]])
    print(shap_values[alts_to_show[1]])
    print(shap_values[alts_to_show[2]])
