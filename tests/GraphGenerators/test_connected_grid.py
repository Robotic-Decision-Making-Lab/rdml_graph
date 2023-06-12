# test_connected_grid.py
# Written Ian Rankin - June 2023
#
# A set of pytests for the connected grid function

import pytest

import rdml_graph as gr
import numpy as np



def test_basic_large_test():
    x_ticks = np.arange(0,200)
    y_ticks = np.arange(0,200)
    map = {}

    G = gr.connected_grid(map, x_ticks, y_ticks)

    assert len(G) == 200*200

def test_simple_full_test():
    x_ticks = [0,1,2]
    y_ticks = [0,1,2]
    map = {}

    G = gr.connected_grid(map, x_ticks, y_ticks)

    for i, n in enumerate(G):
        # corners
        if i == 0 or i == 2 or i == 6 or i == 8:
            assert len(n.e) == 3
        
        # edges
        if i == 1 or i == 3 or i == 5 or i == 7:
            assert len(n.e) == 5
        
        # center
        if i == 4:
            assert len(n.e) == 8


def test_simple_grid_2():
    x_ticks = [0,1,2,3,4,5]
    y_ticks = [0,1,2,3,4,5]
    map = {}

    G = gr.connected_grid(map, x_ticks, y_ticks,grid_size=2)

    for i, n in enumerate(G):
        # corners
        if i == 0 or i == 2 or i == 6 or i == 8:
            assert len(n.e) == 3
        
        # edges
        if i == 1 or i == 3 or i == 5 or i == 7:
            assert len(n.e) == 5
        
        # center
        if i == 4:
            assert len(n.e) == 8






def main():
    x_ticks = [0,1,2,3,4,5,6]
    y_ticks = [0,2,4,6,8]
    map = {}

    G = gr.connected_grid(map, x_ticks, y_ticks)

    print(G[0])
    print(G[1])
    print(G[7])
    print(len(G[7].e))
    print(G[34])


if __name__ == '__main__':
    main()



