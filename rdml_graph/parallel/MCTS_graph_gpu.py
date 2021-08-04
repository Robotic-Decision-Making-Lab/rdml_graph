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
## @file MCTS_graph_gpu.py
# Written Ian Rankin February 2020
# A implementation of MCTS that is modified to parallize the expansion and rollout
# This only works with functions that are fully graph observable.
#
# An implementation of the monte-carlo tree search algorithm
#


import tqdm
import numpy as np
from rdml_graph.mcts import MCTSTree
from rdml_graph.mcts import UCBSelection, randomRollout, bestAvgReward
from rdml_graph.mcts.ParetoFront import ParetoFront
from rdml_graph.parallel.ParallelGraph import get_connection_graph

from numba import cuda

import pdb


## MCTS_graph
# The main entry function to the MCTS algorithm.
# @param start - the entry state of the MCTS algorithm
# @param max_iterations - maximum number of iterations the MCTS algorithm runs.
# @param rewardFunc - the reward function for an end state (sequence, budget, data)
# @oaram budget - the total budget of
# @param selection - selection function (current, budget, data)
# @param rolloutFunc - rollout function (treeState, budget, data)
# @param solutionFunction - (root, data)
# @param data - persistent data across the MCTS.
# @param actor_number - the starting actor number.
# @param multi_obj_dim - the dimension of the multi-objective reward values
#
# @return - solution, reward, opt[root_of_tree]
#           solution - list of states of best path (including start state)
#           reward - float value of best reward.
def MCTS_graph(   start, G, max_iterations, rewardFunc, budget=1.0, selection=UCBSelection, \
            rolloutFunc=randomRollout, solutionFunc=bestAvgReward, data=None, \
            actor_number=0, multi_obj_dim=1, output_tree=False, max_depth=40):
    # Set the root of the search tree.
    root = MCTSTree(start, 0, None)
    #root.unpicked_children = root.successor(budget)

    if multi_obj_dim < -1:
        multi_obj_dim = -multi_obj_dim
        all_values = True
    else:
        all_values = False
    if multi_obj_dim > 1:
        optimal = ParetoFront(multi_obj_dim, alloc_size=int(np.ceil(max_iterations/10)))
    else:
        bestReward = -np.inf
        bestSeq = None

    parallel_data, propogated_paths = setup_multi(G, max_depth)

    # Main loop of MCTS
    for i in tqdm.tqdm(range(max_iterations)):
        try: # Allow keyboard input to interupt MCTS
            current = root

            ######### SELECTION and Expansion
            # Check all possibilties of selection.

            while True:
                if not current.expanded:
                    current.unpicked_children = current.successor(budget)
                    current.expanded = True
                if len(current.unpicked_children) > 0:
                    ######## Expansion
                    # child = current.expandNode()
                    # child.unpicked_children = child.successor(budget)
                    #
                    # current = child

                    # once a node has been successfully expanded break out of selection loop.
                    break
                else:
                    ######## Selection.
                    if len(current.children) <= 0:
                        # reached planning horizon, perform rollout on this node.
                        break

                    current = selection(current, budget, data)
            # end selection expansion while loop.

            #### Parallel expansion and rollout.
            propogated_paths, lengths = \
                start.para_propogate(parallel_data, current.unpicked_children, propogated_paths, budget)




            ######## ROLLOUT
            # perform rollout to the end of a possible sequence.
            sequence = rolloutFunc(current, budget, data)
            rolloutReward, rewardActorNum = rewardFunc(sequence, budget, data)


            if multi_obj_dim > 1:
                optimal.check_and_add(rolloutReward, sequence)
            else:
                if rolloutReward > bestReward:
                    bestReward = rolloutReward
                    bestSeq = sequence
            #pdb.set_trace()

            ######## BACK-PROPOGATE
            current.backpropReward(rolloutReward, rewardActorNum)
        except KeyboardInterrupt:
            break
    # end main for loop

    ######## SOLUTION

    if all_values:
        multi_obj_dim = -multi_obj_dim

        front_rewards, front_paths = optimal.get()
        bestSeq, bestReward = [], 0
        solutionSeq, solutionReward = solutionFunc(root, bestSeq, bestReward, data)
        if output_tree:
            return front_paths, front_rewards, solutionSeq, solutionReward, root
        else:
            return front_paths, front_rewards, solutionSeq, solutionReward
    elif multi_obj_dim > 1:
        front_rewards, front_paths = optimal.get()
        if output_tree:
            return front_paths, front_rewards, root
        else:
            return front_paths, front_rewards
    else:
        solutionSeq, solutionReward = solutionFunc(root, bestSeq, bestReward, data)
        if output_tree:
            return solutionSeq, solutionReward, root
        else:
            return solutionSeq, solutionReward
# End MCTS









def setup_multi(G, max_path_depth=40):
    A, costs, num_edges = get_connection_graph(G)
    pts = np.array([n.pt for n in G], dtype=np.float32)


    threads_per_block = 64
    # max call at once is the size of the threads.
    blocks = A.shape[1] % threads_per_block
    rng_states = cuda.random.create_xoroshiro128p_states( \
            threads_per_block * blocks, seed=1)

    paths = np.ones((A.shape[1], max_path_depth, 2)) * -1
    A = cuda.to_device(A)
    costs = cuda.to_device(costs)
    num_edges = cuda.to_device(num_edges)
    rng_states = cuda.to_device(rng_states)
    pts = cuda.to_device(pts)


    return (A, costs, num_edges, pts, rng_states), paths
