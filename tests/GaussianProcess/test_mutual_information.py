# test_user_gp_probits.py
# Written Ian Rankin - October 2022
#
# Test the probit functions for the user GP

import pytest
import rdml_graph as gr

import numpy as np

import pdb




def test_mutual_info_learner():
    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 0.7), \
                            active_learner= gr.MutualInformationLearner())

    assert gp is not None






def main():
    mu = np.array([0,0,0,0,0,0])
    cov = np.diag([1,1,1,1,1,1])

    Q = [0,4]

    M = 20
    len_Q = len(Q)

    all_w = np.random.multivariate_normal(mu, cov, size=M)
    p = gr.p_human_choice(all_w[:,Q])


    sum_p_over_w = np.sum(p, axis=0)
    info_gain = np.sum(p * np.log2(M * p / sum_p_over_w)) / M

    print(info_gain)



if __name__ == '__main__':
    main()
