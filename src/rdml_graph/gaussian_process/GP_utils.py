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
# GP_utils.py
# Written Ian Rankin - October 2021
#
# A set of utility functions for Gaussian Processes
# Just a spot to put some of the functions, that don't really belong
# anywhere else.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence
import pdb



def get_dk(u, v):
    if (not isinstance(u, (int, float))) or (not isinstance(v, (int, float))):
        raise TypeError("get_dk was not passed a scalar value")
    if u > v:
        return -1
    elif u < v:
        return 1
    else:
        return -1 # probably handle it this way... I could also probably just return 0

## gen_pairs_from_idx
# This function is given the best index selected from a user selection
# and generates the pairs needed to be passed to a preference GP
# @param best_idx - the index that was determined to be best of the given indicies
# @param indicies - the list of indicies the best_idx is better than.
#                   best_idx is allowed to be indicies without breaking anything
#
# @return - list of pairs [(dk, uk, vk), ...]
def gen_pairs_from_idx(best_idx, indicies):
    pairs = []
    for idx in indicies:
        if idx != best_idx:
            pairs.append((get_dk(1,0), best_idx, idx))

    return pairs

## ranked_pairs_from_fake
# generates a all of the ranked pairs from fake inputs
#
def ranked_pairs_from_fake(X, fake_f):
    y = fake_f(X)

    y_sorted_idx = np.argsort(y)

    pairs = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            pairs.append((get_dk(y[i], y[j]), i, j))

    return pairs

## generate_fake_pairs
# generates a set of pairs of data from faked data
# helper function for fake input data
# @param X - the inputs to the function
# @param real_f - the real function to estimate
def generate_fake_pairs(X, real_f, pair_i, data=None):
    Y = real_f(X, data=data)

    pairs = [(get_dk(Y[pair_i], y),pair_i, i) for i, y in enumerate(Y)]
    return pairs





## k_fold_split
# A function to split the datasets for training the PreferenceGP
# @param y - the y data as [[(input data for probit)]]
def k_fold_half(y):
    data = []

    for y_data in y:
        shuffle = np.arange(len(y_data))
        np.random.shuffle(shuffle)

        splits = np.array_split(shuffle, 2)

        # [probits[[split1, split2]], ...]
        split_data = [[y_data[idx] for idx in split] for split in splits]

        data.append(split_data)

    actual_splits = []
    for j in range(2):
        split = []

        for i in range(len(y)):
            split += [np.array(data[i][j])]


        actual_splits.append(split)

    return actual_splits
