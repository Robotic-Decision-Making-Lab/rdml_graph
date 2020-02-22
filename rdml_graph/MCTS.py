# MCTS.py
# Written Ian Rankin February 2020
# Based on code written by Graeme Best, and also code written by Seth McCammon
#
# An implementation of the monte-carlo tree search algorithm
#


import tqdm



# MCTS
# The main entry function to the MCTS algorithm.
def MCTS(start, max_iterations, data=None):



    # main loop of mcts
    for i in tqdm.tqdm(range(max_iterations)):
        current = root

        # SELECTION AND EXPANSION

        pass
