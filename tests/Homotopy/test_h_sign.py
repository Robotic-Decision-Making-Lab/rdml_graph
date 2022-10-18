# test_h_sign.py
# Written Ian Rankin - October 2022
#
# A test suite for testing H signs

import pytest

import rdml_graph as gr
import numpy as np

def test_h_signature():
    # Create HSignature

    x = gr.HomologySignature(10)
    y = gr.HomologySignature(10)
    z = gr.HomologySignature(10)

    assert all(x[5:7] == [0,0])
    assert len(x) == 10
    assert x == y
    assert x[5] == 0

    x.cross(5, 1)
    assert x[5] == 1
    assert x != y

    x.cross(4,-2)
    assert all(x[0:10] == [0,0,0,0,-1,1,0,0,0,0])
    x.cross(5,-1)
    assert all(x[0:10] == [0,0,0,0,-1,0,0,0,0,0])
    x.cross(5, -1)
    assert all(x[0:10] == [0,0,0,0,-1,-1,0,0,0,0])
    x.cross(5, -1)
    assert all(x[0:10] == [0,0,0,0,-1,-1,0,0,0,0])

    t = x.copy()
    assert t == x
    t.cross(3,-1)
    assert t != x
    assert t[3] == -1

    # check to ensure this raises an exception
    with pytest.raises(Exception):
        a = x[11]
