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
# CostmapSamplingFunctions.py
# Written Ian Rankin August 2022
#
# A set of functions to make the planner work with a 2D costmap
# This sample inside of a particular polygon, and then also avoid a particular
# costmap

import numpy as np
import shapely.geometry as geo
from ..core import GeometricNode
from ..core import Edge



## sample points inside of a bounding polygon and outside of obstacles
# @param map - a dictionary with needed parameters for sampling with obstacles
#           bounding - the bounding polygon (shaply)
#           costmap - a numpy 2d array a costmap (indicates obstacle locations)
#           max_free_node - the max value in costmap that is still considered freespace
#           x_ticks
#           y_ticks
# @param num_samples - the total number of points to sample
# @param id_start - the starting point of id's
def sample2DPolygonCostmap(map, num_samples, idStart=0):
    bounding = map['bounding']
    costmap = map['costmap']
    x_ticks = map['x_ticks']
    y_ticks = map['y_ticks']
    max_free = map['max_free_node']
    w_to_img_scale = np.array([x_ticks.shape[0] / (x_ticks[-1] - x_ticks[0]), \
                                    y_ticks.shape[0] / (y_ticks[-1] - y_ticks[0])])

    w_to_img_inter = np.array([x_ticks[0], y_ticks[0]])

    minx, miny, maxx, maxy = bounding.bounds # Gets the bounding box around the actual shape
    scale = np.array([maxx - minx, maxy - miny])
    intercept = np.array([minx, miny])

    points = np.empty((num_samples, 2))

    num_pts = 0
    while num_pts < num_samples:
        # sample point
        pt = np.random.random(2) * scale + intercept

        geoPt = geo.Point(pt[0],pt[1])
        map_pt = (pt - w_to_img_inter) * w_to_img_scale
        cost_at_point = costmap[int(round(map_pt[0])), int(round(map_pt[1]))]
        if bounding.contains(geoPt) and (cost_at_point < max_free) and (cost_at_point >= 0):
            points[num_pts] = pt
            num_pts += 1

    nodes = [GeometricNode(i + idStart, points[i]) for i in range(points.shape[0])]
    return nodes, points


## costmap collision check
# a function to check if the line intersects with an obstacle in the costmap
# @param u - one of the input nodes.
# @param v - the second input node.
# @param map - the input map to check for collisions using.
#           costmap - a numpy 2d array a costmap (indicates obstacle locations)
#           max_free_edge - the max value in costmap that is still considered freespace
#
# @return - True if there is a collision, false otherwise
def costmapCollision(u, v, map):
    costmapCollisionPt(u.pt, v.pt, map)



def costmapCollisionPt(u_pt, v_pt, map):
    costmap = map['costmap']
    x_ticks = map['x_ticks']
    y_ticks = map['y_ticks']
    max_free = map['max_free_edge']
    dist = np.linalg.norm(u_pt - v_pt, ord=2)

    num_pts = dist / ((x_ticks[1]-x_ticks[0])*0.6)
    ts = np.arange(0, 1, 1 / num_pts)
    w_to_img_scale = np.array([x_ticks.shape[0] / (x_ticks[-1] - x_ticks[0]), \
                                    y_ticks.shape[0] / (y_ticks[-1] - y_ticks[0])])

    w_to_img_inter = np.array([x_ticks[0], y_ticks[0]])


    for t in ts:
        #print(t)
        pt = u_pt + t * (v_pt - u_pt)
        map_pt = (pt - w_to_img_inter) * w_to_img_scale


        cost_at_point = costmap[int(round(map_pt[0])), int(round(map_pt[1]))]
        #print('map_pt: ' +str(map_pt) + ' pt: ' + str(pt)+' cost: '+str(cost_at_point))

        # The path is in collision
        if cost_at_point > max_free or (cost_at_point < 0):
            return True


    return False # no found collisions




























#
