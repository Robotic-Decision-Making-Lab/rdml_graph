# TestParetoFront.py
# Written Ian Rankin February 2021
#
# The test Pareto Front.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt

import time

if __name__ == '__main__':
    front = gr.ParetoFront(3, alloc_size=10)



    rewards = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    for i in range(rewards.shape[0]):
        front.check_and_add(rewards[i], i)

    plt.scatter(rewards[:,0], rewards[:, 1], color='red')
    plt.scatter(front.front[:front.size,0], front.front[:front.size,1], color='blue')

    print(front.get())
    plt.show()


    rewards = np.random.random((100, 2))

    for i in range(100):
        if np.sum(rewards[i]) > 1.1:
            rewards[i] = np.array([0,0])

    rewards[0] = np.array([0,0.1])


    front = gr.ParetoFront(2, alloc_size=100)


    start = time.time()

    for i in range(rewards.shape[0]):
        front.check_and_add(rewards[i], i)

    end = time.time()

    print(front.front)
    print(front.front_val)
    print(end - start)

    plt.scatter(rewards[:,0], rewards[:, 1], color='red')
    plt.scatter(front.front[:front.size,0], front.front[:front.size,1], color='blue')
    plt.show()
