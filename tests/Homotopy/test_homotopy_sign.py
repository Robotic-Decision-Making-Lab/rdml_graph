# test_h_sign.py
# Written Ian Rankin - October 2022
#
# A test suite for testing H signs

import pytest

import rdml_graph as gr
import numpy as np

def test_simgple_homotopy_signature():
    # Create HSignature

    sign = gr.HomotopySignature([])

    sign.cross(1, 1)
    assert sign.sign == [1]
    sign.cross(2,1)
    assert sign.sign == [1,2]
    sign.cross(2,-1)
    assert sign.sign == [1]
    sign.cross(2,-1)
    assert sign.sign == [1,-2]
    sign.cross(2,1)
    assert sign.sign == [1]
    sign.cross(2,1)
    assert sign.sign == [1,2]
    sign.cross(2,1)
    assert sign.sign == [1,2,2]

    sign2 = gr.HomotopySignature([-2, -2, 1])

    sign3 = sign + sign2
    assert sign3.sign == [1,1]

    sign4 = sign - sign2
    assert sign4.sign == [1,2,2,-1,2,2]



def test_HEdge():
    n = gr.GeometricNode(0,np.array([-1,1]))
    n1 = gr.GeometricNode(1, np.array([1,2]))
    n2 = gr.GeometricNode(2, np.array([3,4]))
    n3 = gr.GeometricNode(3, np.array([4.4,-3]))
    n4 = gr.GeometricNode(4, np.array([-2.2,-10]))
    n5 = gr.GeometricNode(5, np.array([0,3.2]))


    features = np.array([[0,0], [2,2]])


    e = gr.HEdge(n,n1, gr.HomotopySignature())
    assert e.HSign.sign == []

    e = gr.HEdge(n,n1, gr.HomotopySignature(), features=features)
    assert e.HSign.sign == [1]
    e2 = gr.HEdge(n1,n2, gr.HomotopySignature(), features=features)
    assert e2.HSign.sign == [2]
    e3 = gr.HEdge(n,n2, gr.HomotopySignature(), features=features)
    assert e3.HSign.sign == [1,2]
    e4 = gr.HEdge(n2,n, gr.HomotopySignature(), features=features)
    assert e4.HSign.sign == [-2, -1]


def test_HEdge_equivelence():
    h1 = gr.HomotopySignature([1,4,-5,2])
    h2 = gr.HomotopySignature([1,4,-5,2])
    h3 = gr.HomotopySignature([1,4,5,2])

    assert h1 == h2
    assert h1 != h3

