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
#
# @return output probability distribution of human choice
def p_human_choice(r):
    e = np.exp(r) # exponent of r
    return e / sum(e)



## sample_human_choice
# returns the index of the reward sampled given the probability distribution defined
# by the luce-shepard choice rule.
def sample_human_choice(r, samples=None):
    xk = np.arange(len(r))
    pdf = p_human_choice(r)

    if samples is None:
        return np.random.choice(xk, p=pdf)
    else:
        return np.random.choice(xk, samples, p=pdf)







