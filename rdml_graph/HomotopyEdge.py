# HomotopyEdge.py
# Written Ian Rankin - February 2020
#
# Homotopy augumented edge.
# This is a set of edges defining crossings given the edge.
# Slightly different from Homotopy Node which keeps a copy of the node it is
# representing, but the actual node.
#
# This edge is just an extension of an Edge with extra information to contain
# crossing information.
#
# Based on work by following paper:
# S. Bhattacharya, M. Likhachev, V. Kumar (2012) Topological constraints in
#       search-based robot path planning
#

from rdml_graph import Edge
import numpy as np
from rdml_graph import HSignature

# rayIntersection
# Given a line segment an origin and an angle from the ray, return 0 for
# no intersection, and postive to a crossing in the positive direction, and
# negative for a crossing in the negative direction.
# This code is ported from rdml_utils (Seth McCammon)
# @param pt1 - first point of line segment (numpy 2d)
# @param pt2 - second point of the line segment (numpy 2d)
# @param origin - the origin of the array (numpy 2d)
# @param angle - the angle of the ray. (scalar)
#
# @return - 0 = no intersection, 1 = intersection in positive direction,
#            -1 for negative direction
def rayIntersection(pt1, pt2, origin, angle=0):
    rayDir = np.array([np.cos(angle), np.sin(angle)])

    # Create helper vectors
    v1 = origin - pt1
    v2 = pt2 - pt1
    v3 = np.array([-rayDir[1], rayDir[0]]) # perpendicular to rayDir

    # check for parallel lines segment to ray
    if np.dot(v2, v3) == 0:
        return 0 # no intersection

    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)

    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        # there is an intersection decide direction.
        vPt1 = -v1
        #vPt2 = pt2 - origin

        # x coordinate of a transformed point frame.
        xTran1 = -np.dot(vPt1, v3)
        #xTran2 = np.dot(vPt2, v3)
        if xTran1 < 0:
            return 1 # line segment goes left to right across ray.
        else:
            return -1 # line segment goes right to left across ray.
    else:
        # no intersection
        return 0




class HomotopyEdge(Edge):
    # constructor
    # @param parent - the parent node of the edge.
    # @param child - the child node of the edge.
    # @param num_objects - the total number of topological objects.
    # @param cost - the cost of the edge - defaults to 1
    def __init__(self, parent, child, num_objects=0, cost=1, features=None,ray_angle=0):
        super(HomotopyEdge, self).__init__(parent, child, cost)
        # H-signature fragment (only shows crossing that can occur)

        if features is not None:
            num_objects = features.shape[0]

        self.HSignFrag = np.zeros(num_objects)

        if features is not None:
            self.geo2DHSignCheck(features, ray_angle)


    # geo2DHSignCheck
    def geo2DHSignCheck(self, features, ray_angle=0):
        num_features = len(self.HSignFrag)

        for i in range(num_features):
            self.HSignFrag[i] = rayIntersection(self.p.pt, self.c.pt, \
                                                features[i], ray_angle)
