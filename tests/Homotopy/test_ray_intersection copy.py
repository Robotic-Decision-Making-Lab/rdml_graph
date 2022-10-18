# test_ray_intersection.py
# Written Ian Rankin - October 2022
#
# A test suite for testing intersection with rays.

import pytest

import rdml_graph as gr
import numpy as np

def test_ray_intersections():
    angle = 0 * (np.pi / 180.0)
    origin = np.array([5.0, 5.0])

    pt1 = np.array([5.2, 7.0])
    pt2 = np.array([6.7, -2.0])


    assert gr.rayIntersection(pt1,pt2,origin,angle) == 1
    assert gr.rayIntersection(pt2,pt1,origin,angle) == -1

    assert gr.rayIntersection(pt1,pt2,origin,np.pi/2.0) == 0
    assert gr.rayIntersection(pt2,pt1,origin,np.pi/2.0) == 0
