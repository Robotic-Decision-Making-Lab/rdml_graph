# TestEvaluator.py
# Written Ian Rankin January 2021
#
#

import numpy as np
import rdml_graph as gr

info_field = np.ones((50,50, 2))
eval = gr.MaskedEvaluator(info_field, np.arange(0,50), np.arange(0,50), 9)

path = np.array([[13,11], [13, 34]])

eval.getScore(path)
