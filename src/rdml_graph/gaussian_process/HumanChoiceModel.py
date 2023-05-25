# HumanChoiceModel.py
# Written Ian Rankin - March 2023
#
# A probability model of human decisions given estimated rewards of
# task.
# Using the Luce-Shepard choice rule.
# I'm using the standard formulation define in
# Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh


import numpy as np
from scipy import stats



## p_human_choice
# calculates the probability distribution for a set of estiamted rewards.
# Uses a softmax of the given rewards to model the distribution.
# @param r - the input reward vector (N,)
# @param p - [opt] peakiness of the human choice model, this can be tuned to set how flat or peaky
#              the probability distribution is.
#
# @return output probability distribution of human choice
def p_human_choice(r, p=1.0):
    e = np.exp(r*p) # exponent of r
    if len(r.shape) > 1:
        sum_e = np.sum(e,axis=-1)
        return e / sum_e[:,np.newaxis]
    else:
        return e / np.sum(e)

    



## sample_human_choice
# returns the index of the reward sampled given the probability distribution defined
# by the luce-shepard choice rule.
# @param r - the input reward vector (N,)
# @param p - [opt] peakiness of the human choice model, this can be tuned to set how flat or peaky
#              the probability distribution is.
# @param samples - [opt] sets how many different samples to take.
def sample_human_choice(r, p=1.0, samples=None):
    xk = np.arange(len(r))
    pdf = p_human_choice(r, p=p)

    if samples is None:
        return np.random.choice(xk, p=pdf)
    else:
        return np.random.choice(xk, samples, p=pdf)







