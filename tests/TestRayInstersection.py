# TestRayInstersection.py
# Written Ian Rankin - February 2020
#
# A test suite for testing intersection with rays.

import rdml_graph as gr
import numpy as np



angle = 0 * (np.pi / 180.0)
origin = np.array([5.0, 5.0])

pt1 = np.array([5.2, 7.0])
pt2 = np.array([6.7, -2.0])

correct = True

correct &= gr.rayIntersection(pt1,pt2,origin,angle) == 1
correct &= gr.rayIntersection(pt2,pt1,origin,angle) == -1

correct &= gr.rayIntersection(pt1,pt2,origin,np.pi/2.0) == 0
correct &= gr.rayIntersection(pt2,pt1,origin,np.pi/2.0) == 0

correct &= gr.rayIntersection(pt1,pt2,origin,angle) == -1
correct &= gr.rayIntersection(pt2,pt1,origin,angle) == 1

if correct:
    print('Tests passed!')
else:
    print('A test failed...')
