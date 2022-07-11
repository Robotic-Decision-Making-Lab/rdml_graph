# Copyright 2022 Ian Rankin
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
# SpiralPlanner.py
# Written Ian Rankin
#
# A set of path evaluation functions for information gathering algorithms.

import numpy as np
import pdb

## SpiralPlan
# Generates a spiral plan around a given starting point and radius between points
# This assumes a rectangular search space with bounds, and no obstcales
#
#  ___________________________________________________________
# |                                                           |
# |                                                           |
# |                                                           |
# |                                                           |
# |                                                           |
# |                                                           |
# |                                                       etc |
# |                                           o---------->---o|
# |                                           |  o-------<---o|
# |                                           |  | o----->---o|
# |                                           |  | |  o-->--o |
# |                                           |  | |  | o-o | |
# |                                           |  | |  | X | | |
# |                                           |  | |  o-<-o | |
# |                                           o<-o o----<---o |
#  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#
# @param start - starting location for spiral. Numpy (2)
# @param map - Expects {'x_ticks': numpy array, 'y_ticks': numpy array}
# @param spacing - the spacing between lines in the spiral
#
# @return - numpy array of waypoints (nx2)
def SpiralPlan(start, map, spacing, CW=True):
    dir = 0 # Direction 0=N, 1=E, 2=S, 3=W
    cur_location = start

    # min_x, max_x,
    min_x = start[0]
    max_x = start[0]
    min_y = start[1]
    max_y = start[1]

    # the current largest extent that has been searched
    cur_bound = np.array([max_y, max_x, min_y, min_x])
    bounds = np.array([map['y_ticks'][-1], map['x_ticks'][-1], map['y_ticks'][0], map['x_ticks'][0]])
    axes_for_bound = [1,0,1,0]
    m = np.array([1,1,-1,-1]) # multiplier

    waypoints = [start]

    while np.sum(m*(cur_bound + (m*spacing)) >= m*bounds) < 3 and len(waypoints) < 20:
        new_waypoint = np.copy(waypoints[-1])

        # calculate the next value to udpate
        val = cur_bound[dir] + m[dir] * spacing

        # check if the new point will extend beyond the desired bound.
        # if it does than stop at the bound and reverse the direction of the spiral
        if (m[dir]*val) > (m[dir]*(bounds[dir] - 1.499*m[dir]*spacing)):
            val = bounds[dir] - m[dir]*spacing
            CW = not CW

        # update the waypoint with the new direction
        new_waypoint[axes_for_bound[dir]] = val

        cur_bound[dir] = val

        waypoints.append(new_waypoint)

        # update the direction for the next iteration
        if CW:
            dir += 1
        else:
            dir -= 1
        dir = dir % 4

        # check if spiral is stuck. If it is, remove the selection that made it stuck.
        if np.all(waypoints[-2] == waypoints[-1]):
            #pdb.set_trace()
            waypoints.pop()
            waypoints.pop()
            CW = not CW


    return np.array(waypoints)


































#
