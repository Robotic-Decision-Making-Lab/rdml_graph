# PRM.py
# Written Ian Rankin - February 2020
#
# A set of functions that generate a Probabilistic RoadMap (PRM).
# A very generic version of the PRM is used to generate all other
# types of PRM's.

from ..core import GeometricNode
import numpy as np

def sample2DUniform(map, num_samples):
    samples = np.random.random((num_samples, 2))  * map
    nodes = [GeometricNode(i, samples[i]) for i in range(samples.shape[0])]
    return nodes

def PRM(map, num_points, sampleF=sample2DUniform):
    sample2DUniform(map, num_points)
