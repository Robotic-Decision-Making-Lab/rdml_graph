# TestEvaluator.py
# Written Ian Rankin January 2021
#
#

import numpy as np
import rdml_graph as gr
import time
import matplotlib.pyplot as plt

x_axis = 70
y_axis = 70

info_field = np.ones((x_axis,y_axis, 2)) * np.array([2,3])
eval = gr.CudaEvaluator(info_field, \
                    np.arange(-x_axis,x_axis,2), np.arange(0,y_axis), \
                    radius=10)
evalCPU = gr.MaskedEvaluator(info_field, \
                    np.arange(-x_axis,x_axis,2), np.arange(0,y_axis), \
                    radius=10)

#path = np.array([[3.1,4.2], [8, 5], [13,12], [-10,13], [-15, -5]])

path = np.random.random((20, 2)) * np.array([x_axis, y_axis])



# TODO
reward = eval.getScore(path)
print(reward)
start = time.time()
for i in range(1000):
    reward = eval.getScore(path)
end = time.time()
print("GPU time = " + str(end-start))

start = time.time()
for i in range(1000):
    reward = evalCPU.getScore(path)
end = time.time()
print("CPU time = " + str(end-start))

# gr.plot_multi(path_grid, [path], \
#             x_ticks = np.arange(-x_axis,x_axis,2),\
#             y_ticks = np.arange(0,y_axis),\
#             radius=1.5)
#
# plt.show()



#print(score)
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
