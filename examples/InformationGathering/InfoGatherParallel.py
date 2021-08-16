# Copyright 2021 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# InfoGatherParallel.py
# Written Ian Rankin - March 2020
#
# This example has a random information field generated that the MCTS is ran on
# top of.

import rdml_graph as gr
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pdb


def reward_func(sequence, budget, data):
    evaluator = data[0]

    pts = gr.getWaypoints(sequence)
    score = evaluator.getScore(pts)


    return score, 0

def multi_reward_func(propogated_paths, budget, lengths, data):
    eval = data[0]

    scores = eval.getScoreMulti(propogated_paths, lengths, budget)

    return scores, np.zeros(scores.shape[0], dtype=np.int)



def main():
    dim = 1
    map = {'width': 40, 'height': 40, 'hazards': np.empty((0,2))}

    start = gr.GeometricNode(0, np.array([10, 10]))



    G = gr.PRM(map, 100, 15, connection=gr.HomotopyEdgeConn, initialNodes=[start])


    #### Generate info_field and eval
    #info_field = gr.random_multi_field2d((map['width'], map['height']),\
    #                                        dim, num_gauss=np.array([9,9,9]))
    info_field = gr.random_field2d((map['width'], map['height']), 9)
    # eval = gr.MaskedEvaluator(info_field, \
    #                 np.arange(map['width']), \
    #                 np.arange(map['height']), \
    #                 radius=4)
    x_ticks = np.arange(map['width']) - 20
    y_ticks = np.arange(map['height']) - 20
    field_names = ['1']
    r = 2
    budget = 80
    eval = gr.CudaEvaluator(info_field, \
                    x_ticks, \
                    y_ticks, \
                    budget=budget, \
                    radius=r)


    start_homotopy = gr.HNode(start, \
                              gr.HomotopySignature(), \
                              root=start)

    # generate alternative paths
    alternatives, rewards, data = gr.MCTS_graph(start_homotopy, \
                            G = G, \
                            max_iterations=16000, \
                            budget=budget, \
                            rewardFunc=multi_reward_func, \
                            #selection=gr.paretoUCBSelection, \
                            selection=gr.UCBSelection, \
                            solutionFunc=gr.highestReward, \
                            data=(eval,), \
                            #multi_obj_dim=dim,\
                            output_tree=True)


    print(rewards)
    #print(alternatives)
    # visualize output path
    fig, ps = gr.plot_multi(info_field[:,:,np.newaxis], [alternatives], field_names, legend=True, radius=r,\
                            x_ticks=x_ticks, y_ticks=y_ticks)
    plt.show()








if __name__ == '__main__':
    main()
