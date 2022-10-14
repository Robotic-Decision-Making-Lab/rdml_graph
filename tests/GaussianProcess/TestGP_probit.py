# TestGP_probit.py
# Written Ian Rankin - October 2021
#
# A test function to test GP probits


import rdml_graph as gr
import numpy as np


def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


pp = gr.PreferenceProbit(2.0)

X_train = np.array([0,1,2,3,4.2,6,7])
F = np.array([1,0.5,3,4,5,6,7,2])
F = F / np.linalg.norm(F, ord=np.inf)
pairs = gr.generate_fake_pairs(X_train, f_sin, 0) + \
        gr.generate_fake_pairs(X_train, f_sin, 1) + \
        gr.generate_fake_pairs(X_train, f_sin, 2) + \
        gr.generate_fake_pairs(X_train, f_sin, 3) + \
        gr.generate_fake_pairs(X_train, f_sin, 4)

pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

z = pp.z_k(pairs, F)
print('Z:')
print(z)

print('derivatives (W, dpy_df, py)')
W, dpy_df, py = pp.derivatives(pairs, F)
print('W: ' + str(W))
print('dpy_df: ' + str(dpy_df))
print('py: ' + str(py))
