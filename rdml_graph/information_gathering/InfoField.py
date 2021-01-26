# Copyright 2021 Ian Rankin
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
# InfoField.py
# Written Ian Rankin - January 2021
#
# This set of functions generates various information fields, and handles
# other info field related tasks.

import numpy as np
from scipy.stats import multivariate_normal
from collections import Sequence
import pdb


# random_field
# This function generates a random 2d scalar field using randomly placed
# gaussian spots.
# @param size (width, height in boxes) of the the field
# @param num_gauss - the number of gaussian shapes to use
# @param mean_height - the mean height of the gaussian variables. (exponential distribution)
# @param mean_var - the mean variance of the gaussian dist.. (exponential dist.)
#
# @return numpy(width, height) array
def random_field2d(size, num_gauss=30, mean_height=0.2, mean_var=None):
	field = np.zeros(size)
	if mean_var is None:
		mean_var = float(size[0]) * 0.5

	gausses = []
	mean = np.random.random((num_gauss, 2)) * np.array([size[0], size[1]], dtype=np.float)
	height  = np.random.exponential(scale=mean_height, size=num_gauss)
	var = np.random.exponential(scale=mean_var, size=(num_gauss, 2,2))

	norms = [None] * num_gauss
	height_scale = [0.0] * num_gauss

	for i in range(num_gauss):
		covariance = var[i]
		covariance[1,0] =  0
		covariance[0,1] = 0


		norms[i] = multivariate_normal(mean=mean[i], cov=var[i])
		height_scale[i] = height[i] / norms[i].pdf(mean[i])

	for i in range(size[0]):
		for j in range(size[1]):
			pt = np.array([i,j])
			for k in range(num_gauss):
				field[i,j] += norms[k].pdf(pt) * height_scale[k]


	return field


# force a given sequence or scalar to be the given length
# it fails if the in_seq is not the correct length or a scalar.
# @param in_seq - the input possible sequence
#					(should be either scalar or sequence of correct length)
# @param desired_len - the desired length of the sequence.
def force_seq(in_seq, desired_len):
	if isinstance(in_seq, Sequence) or isinstance(in_seq, np.ndarray):
		if len(in_seq) != desired_len:
			return -1
	else:
		in_seq = [in_seq] * desired_len
	return in_seq


# random_multi_field2d
# This function generates multiple channels of 2d scalar fields.
# @param size - (width, height in pixels) of the the field
# @param dim - the number of multiple objectives
# @param num_gauss - the number of gaussian shapes to use
# @param mean_height - the mean height of the gaussian variables. (exponential distribution)
# @param mean_var - the mean variance of the gaussian dist.. (exponential dist.)
#
# @return numpy(width, height, channels) array
def random_multi_field2d(size, dim=3, num_gauss=30, mean_height=0.2, mean_var=None):
	if mean_var is None:
		mean_var = float(size[0]) * 0.5
	num_gauss = force_seq(num_gauss, dim)
	mean_height = force_seq(mean_height, dim)
	mean_var = force_seq(mean_var, dim)

	field = np.empty((size[0], size[1], dim))

	for i in range(dim):
		field[:,:,i] = random_field2d(size, num_gauss[i], mean_height[i], mean_var[i])

	return field





































#
