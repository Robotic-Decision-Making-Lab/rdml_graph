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

# ActiveLearner.py
# Written Ian Rankin - February 2022
#
# Active learning selection algorithms for Pairwise GP's

import numpy as np
import pdb

## Base Active Learning class.
#
# This class has the needed function to perform active learning from a
# gaussian proccess.
class ActiveLearner:

    def __init__(self):
        self.gp = None

    ## select
    # Selects the given points
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @parm prefer_num - the number of points at the start of the candidates to prefer
    #                   selecting from. (This is designed to be used with pareto-optimal
    #                   selections)
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          value of the best point.
    def select(self, candidate_pts, num_alts, prefer_num=-1):
        raise NotImplementedError('ActiveLearner select is not impleneted')

    ## select_greedy
    # This function greedily selects the best single data point
    # Depending on the selection method, you are not forced to implement this function
    # @param cur_selection - a list of current selections
    # @param data - a user defined tuple of data (determined by the select function)
    #
    # @return the index of the greedy selection.
    def select_greedy(self, cur_selection, data):
        raise NotImplementedError("ActiveLearner select_greey is not implemented as has been called")

    # select_greedy_k
    # This function selects the top k canidates given the data and the select greedy
    # function. This allows a function to select multiple choices in a greedy manner.
    def select_greedy_k(self, cur_selection, num_alts, data):
        num_itr = num_alts - len(cur_selection)


        sel_values = [-float('inf')] * len(cur_selection)

        for i in range(num_itr):
            selection, sel_value = self.select_greedy(cur_selection, data)
            cur_selection.append(selection)
            sel_values.append(sel_value)

        return cur_selection, sel_values

    # select best k
    # this function selects the top k canidate scores and the single best index
    # at each call.
    #
    # @param candidate_scores - numpy array of scores
    # @param num_alts - the number of alternatives to select (includes the highest mean)
    # @param best_idx - the index of the best score that should be the selected path.
    # @parm prefer_num - the number of points at the start of the candidates to prefer
    #                   selecting from. (This is designed to be used with pareto-optimal
    #                   selections)
    #
    # @return indexof [highest_mean, highest_score, next highest score, ...],
    def select_best_k(self, candidate_scores, num_alts, best_idx, prefer_num=-1):
        if prefer_num < 0:
            selected_idx = np.argpartition(candidate_scores, -num_alts)[-num_alts:]
            selected_idx = selected_idx[np.argsort(candidate_scores[selected_idx])][::-1]
        else:
            if prefer_num >= num_alts:
                #pdb.set_trace()
                selected_idx = np.argpartition(candidate_scores[:prefer_num], -num_alts)[-num_alts:]
                selected_idx = selected_idx[np.argsort(candidate_scores[selected_idx])][::-1]
            else:
                selected_idx = np.arange(prefer_num, dtype=np.int)
                selected_idx = selected_idx[np.argsort(candidate_scores[selected_idx])][::-1]
                num_alts = num_alts - prefer_num
                if num_alts > 0:
                    #pdb.set_trace()

                    alt_idx = np.argpartition(candidate_scores[prefer_num:], -num_alts)[-num_alts:]
                    alt_idx += prefer_num
                    alt_idx = alt_idx[np.argsort(candidate_scores[alt_idx])][::-1]

                    selected_idx = np.append(selected_idx, alt_idx, axis=0)

        ######### CHECK if the best candidate is already in the list, if it is move to
        # the front of the list
        found_one_idx = False
        #print(selected_idx)
        #print(candidate_scores[selected_idx])
        for i, idx in enumerate(selected_idx):
            if idx == best_idx:
                found_one_idx = True
                tmp = selected_idx[i]
                selected_idx = np.delete(selected_idx, i)
                selected_idx = np.insert(selected_idx, 0, tmp)
                #selected_idx[i] = selected_idx[0]
                #selected_idx[0] = tmp
                break
        # If the best path is not in the list of selected candidate, then remove worst
        # candidate and put the best at the front.
        if not found_one_idx:
            #worst_idx = np.argmin(canidate[selected_idx])
            worst_idx = len(selected_idx)-1
            selected_idx[worst_idx] = selected_idx[0]
            selected_idx[0] = best_idx

        ######### Sort the other selected indicies to ensure
        # the array looks like this [best_mean, largest_candidate, next_largest_candidate,...]
        selected_idx[1:] = (selected_idx[np.argsort(candidate_scores[selected_idx[1:]])+1])[::-1]


        return selected_idx



class UCBLearner(ActiveLearner):
    ## Constructor
    # @param alpha - the scaler value on the UCB equation UCB = mean + alpha*sqrt(variance)
    def __init__(self, alpha=1):
        super(UCBLearner, self).__init__()
        self.alpha = alpha

    ## select
    # Selects the given points
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @parm prefer_num - the number of points at the start of the candidates to prefer
    #                   selecting from. (This is designed to be used with pareto-optimal
    #                   selections)
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          value of the best point.
    def select(self, candidate_pts, num_alts, prefer_num=-1):
        mu, variance = self.gp.predict(candidate_pts)
        UCB = mu + self.alpha*np.sqrt(variance)

        best_idx = np.argmax(mu)

        selected_idx = self.select_best_k(UCB, num_alts, best_idx, prefer_num)

        return selected_idx, UCB[selected_idx], mu[best_idx]


class DetLearner(ActiveLearner):
    ## Constructor
    # @param alpha - the scaler value on the UCB equation UCB = mean + alpha*sqrt(variance)
    def __init__(self, alpha=1):
        super(DetLearner, self).__init__()
        self.alpha = alpha

    ## select
    # Selects the given points
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @parm prefer_num - the number of points at the start of the candidates to prefer
    #                   selecting from. (This is designed to be used with pareto-optimal
    #                   selections)
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          value of the best point.
    def select(self, candidate_pts, num_alts, prefer_num=-1):
        mu, variance = self.gp.predict(candidate_pts)
        cov = self.gp.cov
        best_idx = np.argmax(mu)

        data = (mu, variance, cov, prefer_num)
        cur_selection = [np.argmax(best_idx)]

        selected_idx, USGV = self.select_greedy_k(cur_selection, num_alts, data)
        print(selected_idx)
        return np.array(selected_idx), USGV, mu[best_idx]


    def select_greedy(self, cur_selection, data):
        mu, variance, cov, prefer_num = data

        best_v = -float('inf')
        best_i = -1

        exp_v = 1.0 / (len(cur_selection) + 1)
        for i in [x for x in range(len(mu)) if x not in cur_selection]:
            idxs = cur_selection + [i]
            sub_grid = np.ix_(idxs, idxs)
            sub_cov = cov[sub_grid]

            GV = np.linalg.det(sub_cov) # Generalized variance.
            SGV = GV ** exp_v # Standardized generalized variance

            value = (1-self.alpha)*mu[i] + self.alpha*SGV
            #print('i: ' + str(i)+ ' SGV: ' + str(SGV) + ' value: ' +str(value))

            if value > best_v:
                best_v = value
                best_i = i

        return best_i, value









#
