# MutualInformationLearner.py
# Written Ian Rankin - March 2023
#
# A set of code to select active learning algorithms for user preferences.
# This code implements the mutual information model described in 
# Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh
#
# Potentially with modifications



from rdml_graph.gaussian_process import ActiveLearner
import numpy as np


class MutualInformationLearner(ActiveLearner):

    ## constructor for the mutual information learner.
    def __init__(self):
        super(MutualInformationLearner, self).__init__()
        self.M = 10 # random value at the moment

    

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
        # Need to define p(q|Q,w)

        # only need
        # p(q|w,Q) human choice model given w = rewards and Q is the particular query.
        
        # shouldn't this need p(w) as well? No because it is sampled from the distribution
        # Can I solve that exactly with the GP?



        
        pass









