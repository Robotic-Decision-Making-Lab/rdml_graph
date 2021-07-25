# gis_utilities.py
# Written Ian Rankin - July 2021
#
# A set of utility function for handling GIS functions.

import haversine as hv
import numpy as np


## haversine
# computes the haversine distance between two points.
# This is just a wrapper around the haversine function from the package haversine
# @param pt1 - a lat-lon coordinate as a tuple, list, or numpy array (lon, lat)
# @param pt2 - a lat-lon coordinate as a tuple, list, or numpy array (lon, lat)
#
# @return the distance between pt1 and pt2 in km
def haversine(pt1, pt2):
    if isinstance(pt1, list) or isinstance(pt1, tuple) or isinstance(pt1, np.array):
        p1 = (pt1[1], pt1[0])
    else:
        raise TypeError("haversine function not given unknown type for pt1: " + str(type(pt1)))

    if isinstance(pt2, list) or isinstance(pt2, tuple) or isinstance(pt2, np.array):
        p2 = (pt2[1], pt2[0])
    else:
        raise TypeError("haversine function not given unknown type for pt2: " + str(type(pt2)))

    distance = hv.haversine(p1, p2)
    return distance


## deg_lon_to_km
# calculates the length of one degree longitude at a particular latitude.
# @param lat - the lattitude to check length of 1 degree of longitude (float)
#
# @return the km of one degree of longitude
def deg_lon_to_km(lat):
    return haversine((0, lat), (1, lat))

## deg_lat_to_km
# return the length of one degree of latitude (equal across all longitudes)
# @return the length of one degree of latitude
def deg_lat_to_km():
    return haversine((0,0), (0,1))
