# Copyright 2020 Ian Rankin
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
solution, reward, data = gr.MCTS(start, 500, rewardDumb, budget=0.5, solutionFunc=gr.highestReward)

print('Solution reward = ' + str(reward))
for state in solution:
    print(state)
