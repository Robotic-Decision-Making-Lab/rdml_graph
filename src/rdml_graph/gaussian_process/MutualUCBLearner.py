# MutualUCBLearner.py
# Written Ian Rankin - April 2023
#
# A set of code to select queries for user preferences.
# This builds off of mutual information model described in 
# [1] Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh
#
#  This method only needs
# p(q|w,Q) human choice model given w = rewards and Q is the particular query.
# by sampling from the distribution of potential weights. 
#
# However, since mutual information doesn't perform Bayesian optimization combine it
# with an UCB to provide an exploration and exploitation term.
#


from rdml_graph.gaussian_process import ActiveLearner
from rdml_graph.gaussian_process import MutualInformationLearner

import numpy as np


class MutualUCBLearner(MutualInformationLearner):

    ## constructor for the mutual information learner.
    def __init__(self, alpha = 0.5):
        super(MutualUCBLearner, self).__init__()
        self.alpha = alpha
    

    # ## select
    # # Selects the given points
    # # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # # @param num_alts - the number of alterantives to selec (including the highest mean)
    # # @parm prefer_num - the number of points at the start of the candidates to prefer
    # #                   selecting from. (This is designed to be used with pareto-optimal
    # #                   selections)
    # #
    # # @return [highest_mean, highest_selection, next highest selection, ...],
    # #          selection values for candidate_pts,
    # #          value of the best point.
    # def select(self, candidate_pts, num_alts, prefer_num=-1):
    #     mu, variance = self.gp.predict(candidate_pts)
    #     cov = self.gp.cov
    #     # sample M possible parameters w (reward values of the GP)
    #     all_w = np.random.multivariate_normal(mu, cov, size=self.M)
        

    #     data = (mu, all_w,)
    #     cur_selection = [np.argmax(mu)]

    #     selected_idx = []
    #     mutual_ucbs = []
    #     scores = []
    #     for i in range(num_alts):
    #         cur_sel_idx, mutual_ucb = self.select_greedy(cur_selection, data)

    #         selected_idx.append(cur_sel_idx)
    #         cur_selection.append(cur_sel_idx)
    #         mutual_ucbs.append(mutual_ucb)
    #         scores.append(mu[cur_sel_idx])


    #     #print(selected_idx)
    #     return np.array(selected_idx), np.array(mutual_ucbs), np.array(scores)


    # @override
    ## select_greedy
    # This function greedily selects the best single data point
    # Depending on the selection method, you are not forced to implement this function
    # @param cur_selection - a list of current selections
    # @param data - a user defined tuple of data (determined by the select function)
    #
    # @return the index of the greedy selection.
    def select_greedy(self, cur_selection, data):
        mu, all_w = data

        best_v = -float('inf')
        best_i = -1

        for i in [x for x in range(len(mu)) if x not in cur_selection]:
            idxs = cur_selection + [i]
            info_gain = self.calc_info_gain(idxs, all_w)
            value = (self.alpha * mu[i]) + ((1 - self.alpha) * info_gain)

            # print('i: ' + str(i))
            # print('info_gain: ' + str(info_gain))
            # print('mu: ' + str(mu[i]))
            # print('value: ' + str(value))

            if value > best_v:
                best_v = value
                best_i = i

        return best_i, best_v





