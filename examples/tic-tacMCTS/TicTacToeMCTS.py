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
# TicTacToeMCTS.py
# Written Ian Rankin February 2020
#
# This code is an example usage of the MCTS algorithm implementation used in
# this software package on a simple toy problem.

import rdml_graph as gr
import numpy as np
import copy
import pdb

class TicTacToeState(gr.State):
    def __init__(self, agentTurn=0, previousState=np.zeros((3,3))):
        self.board = copy.copy(previousState)
        # agentTurn = 0 1's, agentTurn = 1 -1's
        self.agentTurn = agentTurn

    # successor function for the tic-toe game.
    # @return list with next state, cost (0, unless in final game state), and
    #    the agent for the turn returned.
    def successor(self):
        result = []
        if self.winState() != -1:
            return result
        nextTurn = 1 - self.agentTurn
        token = 1 if self.agentTurn==0 else -1

        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    nextState = TicTacToeState(agentTurn=nextTurn, previousState=self.board)
                    nextState.board[x][y] = token
                    if nextState.winState() != -1:
                        # state, cost, agent
                        result.append((nextState, 1, self.agentTurn))
                    result.append((nextState, 0, self.agentTurn))
        return result

    def getLabel(self, data=None):
        return str(self.board)

    # winState
    # indicates if the state is currently in a winning state
    # return - 0 for agent 0 winning, 1 for agent 1, 2 - for cat's game and -1 for not a winning state.
    def winState(self):
        #print(self.board)
        #pdb.set_trace()

        # vertical checks
        win = 0
        for x in range(3):
            first = self.board[x][0]
            if first != 0:
                localWin = True
                for y in range(1,3):
                    if self.board[x][y] != first:
                        localWin = False
                        break
                if localWin:
                    win = first
                    break

        if win == 0:
            # check horizontals
            for y in range(3):
                first = self.board[0][y]
                if first != 0:
                    localWin = True
                    for x in range(1,3):
                        if self.board[x][y] != first:
                            localWin = False
                            break
                    if localWin:
                        win = first
                        break

        if win == 0:
            # check diagonals
            diagonals = [[[0,0],[1,1],[2,2]], [[0,2], [1,1], [2,0]]]

            for diag in diagonals:
                first = self.board[diag[0][0]][diag[0][1]]
                if first != 0:
                    localWin = True
                    for index in diag:
                        if self.board[index[0]][index[1]] != first:
                            localWin = False
                            break
                    if localWin:
                        win = first
                        break
        if win != 0:
            # check which agent won.
            if win == -1:
                return 1
            else:
                return 0
        else:
            # check for cat's game
            for x in range(3):
                for y in range(3):
                    if self.board[x][y] == 0:
                        return -1 # not a cat's game!
            return 2 # cats game!


# reward function for tic tac toe
def ticTacToeReward(listOfStates, budget, data):
    lastState = listOfStates[-1]

    winning = lastState.winState()
    if winning == 0:
        return 1, 0 # reward = 1, agent num = 0
    elif winning == 1:
        return 1, 1 # reward = 1, agent num = 1
    else:
        return 0, 0 # reward = 0, doesn't matter which agent gets reward of 0


start = TicTacToeState(agentTurn=1)
start.board[1][1] = 1.0

solution, reward, data = gr.MCTS(start, 50, ticTacToeReward, actor_number=0, \
                            solutionFunc=gr.mostSimulations, output_tree=True)
root = data['root']
t = root.get_viz(labels=True)
t.view()

print('Solution reward = ' + str(reward))
for state in solution:
    print(state.board)

#print(solution)
