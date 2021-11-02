# TestEvaluator.py
# Written Ian Rankin January 2021
#
#

import numpy as np
import matplotlib.pyplot as plt
import rdml_graph as gr

import pdb

x_axis = 40
y_axis = 40


path = np.array([[3.1,4.2], [8, 5], [13,12], [-10,13], [-15, -5], [15, -5]])

budget = 0
for i in range(1, path.shape[0]):
    budget += np.linalg.norm(path[i] - path[i-1], ord=2)

info_field = np.ones((x_axis,y_axis, 2)) * np.array([2,3])
x_ticks = np.arange(-x_axis/2,x_axis/2)
y_ticks = np.arange(-y_axis/2,y_axis/2)
eval = gr.MaskedEvaluator(info_field, \
                    x_ticks, y_ticks,
                    radius=2, \
                    budget=budget)



score, mask = eval.getScore(path, return_mask=True)

#plt.matshow(mask)
print(mask[np.newaxis,:,:].shape)
gr.plot_multi(mask[:,:, np.newaxis], [path], x_ticks=x_ticks, y_ticks=y_ticks)

print(score)

plt.show()

#

# info_field = np.ones((x_axis, y_axis))
# step_size = 3
# x_ticks = np.arange(25.5, 25.5+step_size*x_axis, step_size)
# y_ticks = np.arange(3.7, 3.7+step_size*y_axis, step_size)
#
#
# eval = gr.MaskedEvaluator(info_field, x_ticks, y_ticks, radius=4)
#
# path = np.array([[34.5, 14.2], [56.3, 23], [56.3, 50], [101, 26.3]])
#
# score = eval.getScore(path)
# print(score)
