# Copyright 2023 Ian Rankin
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

# PreferenceModel.py
# Written Ian Rankin - November 2023
#
# Base set of code for adding preference data.
# 

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence
from rdml_graph.gaussian_process import get_dk


class PreferenceModel():

    # init function to setup all needed varaibles
    # @param 
    def __init__(self, pareto_pairs=False, other_probits={}):
        self.optimized = False

        self.pareto_pairs = pareto_pairs
        self.probit_idxs = {'relative_discrete': 0}

        i = 1
        for key in other_probits:
            self.probit_idxs[key] = i
            i += 1

        self.y_train = [None] * len(self.probit_idxs)
        self.X_train = None

        self.prior_idx = None


    ## add_prior
    # this function adds prioir data to the GP if desired. Desigend to work with
    # the pareto_pairs constraint to generate a function that ensures pareto_pairs
    # @param bounds - the bounds for the prior pts numpy array (nxn)
    # @param num_pts - the number of prior pts to add
    def add_prior(self, bounds = np.array([[0,1],[0,1]]), num_pts = 100, \
                    method='random', pts=None):
        if self.pareto_pairs == False:
            print('Asked to add prior information without setting pareto pairs to be used. Not adding prior points')
            return

        scaler = bounds[:,1] - bounds[:,0]
        bias = bounds[:,0]

        if method == 'random':
            pts = np.random.random((num_pts, bounds.shape[0])) * scaler + bias

            # replace 2 of the points with the min a max of the prior bounds
            if num_pts > 2:
                pts[0] = bounds[:,0]
                pts[1] = bounds[:,1]

            print(pts)

        elif method == 'exact':
            pts = pts
            num_pts = pts.shape[0]

        if self.X_train is not None:
            self.prior_idx = (self.X_train.shape[0], self.X_train.shape[0]+num_pts)
        else:
            self.prior_idx = (0, num_pts)
        self.add(pts, [], type='relative_discrete')
        self.remove_without_reference()



    ## This function removes all training points with no references
    # This is used because prior points can have no references and cause problems
    # during optimization because of it.
    #
    # @post - X_train has removed indicies, all references in y_train have been
    #          decremented
    def remove_without_reference(self, remove_prior=True):
        counts = np.zeros(self.X_train.shape[0])

        # iterate through each type of training data
        for type in self.probit_idxs:
            y = self.y_train[self.probit_idxs[type]]
            if type == 'relative_discrete':
                for pair in y:
                    counts[pair[1]] += 1
                    counts[pair[2]] += 1

        # check which pts don't have any counts
        #idx_to_rm = [x for x in range(len(counts)) if counts[x] == 0]
        idx_to_rm = []
        cur_cts = 0
        for i in range(len(counts)):
            if counts[i] == 0:
                idx_to_rm.append(i)
                cur_cts += 1

            counts[i] = cur_cts

        # remove X_train points to remove
        self.X_train = np.delete(self.X_train, idx_to_rm, axis=0)

        # reduce indicies of y_train to match removed indicies
        for type in self.probit_idxs:
            y = self.y_train[self.probit_idxs[type]]
            if type == 'relative_discrete':
                for pair in y:
                    pair[1] -= counts[pair[1]]
                    pair[2] -= counts[pair[2]]


        if remove_prior:
            # update the index if they have been removed
            prior_idx = (self.prior_idx[0], self.prior_idx[1] - len(idx_to_rm))
            self.prior_idx = prior_idx



    ## get_prior_pts
    # get the set of prior points if they exist
    # @return numpy array of X_train if it exists, None otherwise
    def get_prior_pts(self):
        if self.prior_idx is not None:
            return self.X_train[self.prior_idx[0]:self.prior_idx[1]]
        else:
            return None

    ## reset
    # This function resets all points for the GP
    def reset(self):
        self.y_train = [None] * len(self.probit_idxs)
        self.X_train = None
        self.prior_idx = None

    ## add_training
    # adds training data to the gaussian process
    # appends the data if there already is some training data
    # @param X - the input training data
    # @param y - list of discrete pairs [(dk, uk, vk), ...]
    #                       dk = -1 if u > v, dk = 1 if v > u
    #                       uk = index of the input (for the input set of points)
    #                       vk = index of the second input
    #                       @NOTE That this function updates the indicies if there
    #                       is already training data.
    # @param training_sigma - [opt] sets the uncertianty in the training data
    #                          accepts scalars or a vector if each sample has
    #                          a different uncertianty.
    def add(self, X, y, type='relative_discrete', training_sigma=0):
        if not isinstance(training_sigma, Sequence):
            training_sigma = np.ones(len(y)) * training_sigma

        if self.X_train is None:
            self.X_train = X
            len_X = 0
        else:
            len_X = len(self.X_train)
            self.X_train = np.append(self.X_train, X, axis=0)

        if type == 'relative_discrete':
            if y == []:
                pass
            elif self.y_train[self.probit_idxs[type]] is None:
                self.y_train[self.probit_idxs[type]] = np.array(y)
            else:
                # reset index of pairwise comparisons
                y = [(d, u+len_X, v+len_X) for d, u, v in y]

                self.y_train[self.probit_idxs[type]] = \
                    np.append(self.y_train[self.probit_idxs[type]], np.array(y), axis=0)
        elif type == 'ordinal':
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if len(y.shape) == 1:
                new_y = np.empty((y.shape[0], 2), dtype=int)
                new_y[:,0] = y
                new_y[:,1] = np.arange(0, y.shape[0])
            else:
                new_y = y
            if self.y_train[self.probit_idxs[type]] is None:
                self.y_train[self.probit_idxs[type]] = new_y
            else:
                self.y_train[self.probit_idxs[type]] = \
                    np.append(self.y_train[self.probit_idxs[type]], new_y, axis=0)
        elif type == 'abs':
            if isinstance(y, tuple):
                v = y[0]
                idxs = y[1]
            elif isinstance(y, np.ndarray):
                v = y
                idxs = np.arange(len_X, y.shape[0]+len_X)
            else:
                print('abs type received unknown type for y')
                return

            if self.y_train[self.probit_idxs[type]] is not None:
                v = np.append(self.y_train[self.probit_idxs[type]][0], v, axis=0)
                idxs = np.append(self.y_train[self.probit_idxs[type]][1], idxs, axis=0)

            self.y_train[self.probit_idxs[type]] = (v, idxs)


        if self.pareto_pairs:
            pairs = []
            d_better = get_dk(1,0)
            # Go through each new sample and check if it pareto optimal to others
            for i, x in enumerate(X):
                dominate = np.all(x > self.X_train, axis=1)

                cur_pairs = [(d_better, i+len_X, j) for j in range(len(dominate)) if dominate[j]]
                pairs += cur_pairs

            if self.y_train[self.probit_idxs['relative_discrete']] is None:
                self.y_train[self.probit_idxs['relative_discrete']] = np.array(pairs)
            else:
                # only add pairs if there is any pareto pairs to add.
                if len(pairs) > 0:
                    self.y_train[self.probit_idxs['relative_discrete']] = \
                        np.append(self.y_train[self.probit_idxs['relative_discrete']], \
                                    np.array(pairs), axis=0)
        # end if for pareto_pairs


        self.optimized = False


    # calculate the log_likelyhood of the training data.
    # log p(Y|F)
    def log_likelyhood_training(self, F):
        log_p_w = 0.0
        for j, probit in enumerate(self.probits):
            if self.y_train[j] is not None:
                p_w_local = probit.log_likelihood(self.y_train[j], F)

                log_p_w += p_w_local

        return log_p_w

    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        raise NotImplementedError("PreferenceModel optimize function not implemented")
        self.optimized = True

