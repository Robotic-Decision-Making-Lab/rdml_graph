# TestMCTSSingle.py
# Written Ian Rankin - March 2020
#
# Testing a bug where a single action results in a longer path.

import rdml_graph as gr

class IncrementState(gr.State):
    def __init__(self, num):
        self.num = num

    def successor(self):
        return [(IncrementState(self.num+1), 1)]

    def __str__(self):
        return 'IncState(' + str(self.num) + ')'

def rewardDumb(sequence, budget, data):
    return 1, 0


start = IncrementState(0)
solution, reward = gr.MCTS(start, 500, rewardDumb, budget=0.5, solutionFunc=gr.highestReward)

print('Solution reward = ' + str(reward))
for state in solution:
    print(state)
