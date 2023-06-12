# MutualInformationLearner.py
# Written Ian Rankin - March 2023
#
# A set of code to select active learning algorithms for user preferences.
# This code implements the mutual information model described in 
# [1] Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh
#
#  This method only needs
# p(q|w,Q) human choice model given w = rewards and Q is the particular query.
# by sampling from the distribution of potential weights. 
#
# shouldn't this need p(w) as well? No because it is sampled from the distribution
# Can I solve that exactly with the GP?



from rdml_graph.gaussian_process import ActiveLearner
from rdml_graph.gaussian_process import p_human_choice
import numpy as np

from copy import copy

import pdb

class MutualInformationLearner(ActiveLearner):

    ## constructor for the mutual information learner.
    def __init__(self, fake_func=None):
        super(MutualInformationLearner, self).__init__()
        self.M = 50 # random value at the moment
        self.peakiness = 10
        self.fake_func = fake_func

    
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
        # sample M possible parameters w (reward values of the GP)
        all_w = np.random.multivariate_normal(mu, cov, size=self.M)

        if self.fake_func is not None:
            fake_f_mean = np.mean(self.fake_func(candidate_pts))
            samp_mean = np.mean(all_w)

            all_w = all_w * (fake_f_mean / samp_mean)        

        data = (mu, all_w)
        cur_selection = [np.argmax(mu)]

        selected_idx = []
        info_gains = []
        scores = []
        for i in range(num_alts):
            cur_sel_idx, info_gain = self.select_greedy(cur_selection, data)

            selected_idx.append(cur_sel_idx)
            cur_selection.append(cur_sel_idx)
            info_gains.append(info_gain)
            scores.append(mu[cur_sel_idx])


        #print(selected_idx)
        return np.array(selected_idx), np.array(info_gain), np.array(scores)


    ## select_previous
    # Selects the active learning selections from the canidate pts with a list of previous
    # selections. Particuarly useful for pairwise active learning methods.
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param prev_selection - a list of indicies in candidate pts of previously selected paths
    # @param num_alts - [opt] the number of alternative points to select from.
    # @param prefer_num - [opt] the number of points at the start of the candidates to prefer
    #                   selecting from. (This is designed to be used with pareto-optimal
    #                   selections)
    #
    #
    # @return - index of the selection
    def select_previous(self, candidate_pts, prev_selection=[], num_alts=1, prefer_num=-1):
        mu, variance = self.gp.predict(candidate_pts)
        # this is making the assumption that the model has a covariance
        cov = self.gp.cov


        # need to calculate the information gain given a set of possible
        # parameter models.
        

        # sample M possible parameters w (reward values of the GP)
        all_w = np.random.multivariate_normal(mu, cov, size=self.M)
        
        if self.fake_func is not None:
            fake_f_mean = np.mean(self.fake_func(candidate_pts))
            samp_mean = np.mean(all_w)

            print('Scaling using fake function: ' + str(fake_f_mean / samp_mean))
            all_w = all_w * (fake_f_mean / samp_mean)


        data = (mu, all_w)
        cur_selection = copy(prev_selection)

        selected_idx = []
        for i in range(num_alts):
            cur_sel_idx, info_gain = self.select_greedy(cur_selection, data)

            selected_idx.append(cur_sel_idx)
            cur_selection.append(cur_sel_idx)

        #print(selected_idx)
        return np.array(selected_idx)
        
        

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
            value = self.calc_info_gain(idxs, all_w)

            if value > best_v:
                best_v = value
                best_i = i

        return best_i, best_v




        

    # calculate the info gain for a query Q given the sampled parameters / reward W
    # only need p(q|w,Q) human choice model given w = rewards and Q is the particular query.
    # shouldn't this need p(w) as well? No because it is sampled from the distribution
    # Can I solve that exactly with the GP?
    #
    # @param Q - list of indicies of query.
    # @param all_w - a matrix of possible rewards for sample set of parameters [M,N]
    #                   M - number of samples
    #                   N - dimension of candidate points.
    #

    def calc_info_gain(self, Q, all_w):
        # Find the probabilities of human selecting a query given the possible reward values
        p = p_human_choice(all_w[:,Q], self.peakiness)
        # find the sum of the probabilities of w
        sum_p_over_w = np.sum(p, axis=0)

        #pdb.set_trace()

        # Find the information gain using the sample equation (4) in [1]
        info_gain = np.sum(p * np.log2(self.M * p / sum_p_over_w)) / self.M

        return info_gain

