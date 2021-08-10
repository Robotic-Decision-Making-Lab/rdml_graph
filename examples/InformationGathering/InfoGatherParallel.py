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


def reward_func(sequence, budget, data):
    evaluator = data[0]

    pts = gr.getWaypoints(sequence)
    score = evaluator.getScore(pts)


    return score, 0

def multi_reward_func(propogated_paths, budget, lengths, data):
    eval = data[0]

    scores = eval.getScoreMulti(propogated_paths, lengths, budget)

    return scores, 0



def main():
    dim = 3
    map = {'width': 40, 'height': 40, 'hazards': np.empty((0,2))}

    start = gr.GeometricNode(0, np.array([10, 10]))



    G = gr.PRM(map, 100, 6.0, connection=gr.HomotopyEdgeConn, initialNodes=[start])


    #### Generate info_field and eval
    info_field = gr.random_multi_field2d((map['width'], map['height']),\
                                            dim, num_gauss=np.array([5,7,9]))
    # eval = gr.MaskedEvaluator(info_field, \
    #                 np.arange(map['width']), \
    #                 np.arange(map['height']), \
    #                 radius=4)
    eval = gr.CudaEvaluator(info_field, \
                    np.arange(map['width']), \
                    np.arange(map['height']), \
                    radius=4)


    start_homotopy = gr.HNode(start, \
                              gr.HomotopySignature(), \
                              root=start)

    # generate alternative paths
    alternatives, rewards, data = gr.MCTS_graph(start_homotopy, \
                            G = G, \
                            max_iterations=400, \
                            budget=50, \
                            rewardFunc=multi_reward_func, \
                            selection=gr.paretoUCBSelection, \
                            data=(eval,), \
                            multi_obj_dim=dim,\
                            output_tree=True)


    # visualize output path









if __name__ == '__main__':
    main()
