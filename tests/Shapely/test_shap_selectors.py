# test_shap_selectors.py
# Written Ian Rankin - Dec 2022
#
#

import pytest

import rdml_graph as gr
import numpy as np



def test_shap_similar_except():
    vals = np.array([[0.3, 0.6, 0.1],\
                     [0.5, 0.3, 0.2], \
                     [0.1, 0.1, 0.8], \
                     [0.4, 0.4, 0.2], \
                     [0.2, 0.3, 0.5]])

    alts_to_show, features = gr.select_alts_from_shap_diff(0, vals, 2, 'similar_except')

    assert alts_to_show[0] == 0
    assert features[0] == 1
    assert alts_to_show[1] == 1


def test_shap_prefer_pareto():
    vals = np.array([[0.3, 0.6, 0.1],\
                     [0.5, 0.3, 0.2], \
                     [0.1, 0.1, 0.8], \
                     [0.4, 0.4, 0.2], \
                     [0.2, 0.3, 0.5]])

    pareto_idx = [0,1]

    alts_to_show, features = gr.select_alts_from_shap_diff(0, vals, 2, 'prefer_pareto', pareto_idx=pareto_idx)

    assert alts_to_show[0] == 0
    assert features[0] == 1
    assert alts_to_show[1] == 1
