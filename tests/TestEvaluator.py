# TestEvaluator.py
# Written Ian Rankin January 2021
#
#

import numpy as np
import rdml_graph as gr

x_axis = 20
y_axis = 15

info_field = np.ones((x_axis,y_axis, 2)) * np.array([2,3])
eval = gr.MaskedEvaluator(info_field, \
                    np.arange(0,x_axis), np.arange(0,y_axis), \
                    radius=1.5)

path = np.array([[3.1,4.2], [8, 5], [13,12], [21,13]])

# score = eval.getScore(path)
#
# print(score)

info_field = np.ones((x_axis, y_axis))
step_size = 3
x_ticks = np.arange(25.5, 25.5+step_size*x_axis, step_size)
y_ticks = np.arange(3.7, 3.7+step_size*y_axis, step_size)


eval = gr.MaskedEvaluator(info_field, x_ticks, y_ticks, radius=5)

path = np.array([[29.7, 14.2], [56.3, 23], [56.3, 9]])

score = eval.getScore(path)
print(score)
