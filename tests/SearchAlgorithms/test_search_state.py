# test_search_state.py
# Written Ian Rankin - October 2022
#
#

import pytest

import rdml_graph as gr
import heapq
import numpy as np

def test_search_state_init():
    z = gr.State()
    x = gr.SearchState(state=z, rCost=4)

    assert x.rCost == 4
    assert type(x) is gr.SearchState

def test_search_state_heap():
    z = gr.State()
    x = gr.SearchState(state=z, rCost=4) # parent state
    heap = []

    for i in range(15):
        r = np.random.random()
        s = gr.SearchState(state=gr.State(),rCost=r, parent=x)
        s.invertCmp = True
        heapq.heappush(heap, s)


    prev_state = heapq.heappop(heap)
    
    while len(heap) > 0:
        state = heapq.heappop(heap)

        assert state.rCost < prev_state.rCost

        prev_state = state

def test_get_path():
    z = gr.State()
    x = gr.SearchState(state=z, rCost=4) # parent state
    heap = []

    for i in range(3):
        r = np.random.random()
        s = gr.SearchState(state=gr.State(),rCost=r, parent=x)
        s.invertCmp = True
        heapq.heappush(heap, s)

    p = heap[-1].getPath()

    assert len(p) == 2
    assert(p[0] == z)
    assert(p[1] != z)

def test_get_rev_path():
    z = gr.State()
    x = gr.SearchState(state=z, rCost=4) # parent state
    heap = []

    for i in range(3):
        r = np.random.random()
        s = gr.SearchState(state=gr.State(),rCost=r, parent=x)
        s.invertCmp = True
        heapq.heappush(heap, s)

    p = heap[-1].getRevPath()

    assert len(p) == 2
    assert(p[1] == z)
    assert(p[0] != z)

