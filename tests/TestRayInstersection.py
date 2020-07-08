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
