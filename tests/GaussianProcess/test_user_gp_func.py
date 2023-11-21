# test_user_gp_func.py
# Written Ian Rankin November 2023
#
#

import pytest

import rdml_graph as gr
import numpy as np

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_loss_f():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = gr.generate_fake_pairs(X_train, f_sin, 0) + \
            gr.generate_fake_pairs(X_train, f_sin, 1) + \
            gr.generate_fake_pairs(X_train, f_sin, 2) + \
            gr.generate_fake_pairs(X_train, f_sin, 3) + \
            gr.generate_fake_pairs(X_train, f_sin, 4)


    gp = gr.PreferenceGP(gr.RBF_kern(0.5, 0.7))
    gp.add(X_train, pairs)

    l_f = gp.loss_F(np.random.random(len(X_train)))

    assert l_f is not None
    assert isinstance(l_f, float)
    assert not np.isnan(l_f)

