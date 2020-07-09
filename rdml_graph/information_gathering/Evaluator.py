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
# Evaluator.py
# Written Seth McCammon, revised Ian Rankin April 2020
#
# A set of path evaluation functions for information gathering algorithms.


import pdb, itertools, time, sys
from scipy.interpolate import RegularGridInterpolator
import numpy as np
#from rdml_utils import locationArange, Location, euclideanDist, HSignature

from shapely.geometry import Point, LineString



import matplotlib.pyplot as plt



class PathEvaluator(object):
    """PathEvaluator default path evaluator class for information gathering tasks"""
    def __init__(self):
        pass

    # overide
    # getScore, gets the score of path given within the budget
    # @param path - the path as 2d numpy array (n x 2)
    # @param budget - the budget of the path, (typically path length)
    def getScore(self, path, budget=float('inf')):
        print('OOPS PathEvaluator function did not get overidden exiting program')
        sys.exit()


class PathEvaluatorWithRadius(PathEvaluator):
  """docstring for PathEvaluatorWithRadius"""
  def __init__(self, info_field, cost_field, x_ticks, y_ticks, yaml_eval):
    # self.rep_pts = rep_pts
    #self.reference_h_sig = HSignature(())
    self.info_field = info_field
    self.cost_field = cost_field
    self.radius = yaml_eval['radius']
    self.method = yaml_eval['method']
    self.step_size = self.radius / 10.
    self.pts = [Point([x, y]) for x, y in itertools.product(x_ticks, y_ticks)]
    # self.xx, self.yy = np.meshgrid(x_ticks, y_ticks)
    self.yy, self.xx = np.meshgrid(y_ticks, x_ticks)
    self.x_ticks = x_ticks
    self.y_ticks = y_ticks

  def getExactMask(self, path, budget=float('inf')):
    budgeted_path, budgeted_path_len = applyBudget(path, budget)
    mask = np.zeros(self.info_field.shape)

    if isinstance(budgeted_path, np.ndarray):
      #shapely_path = [Location(xlon=pt[0], ylat=pt[1]).shapelyPoint() for pt in budgeted_path]
      shapely_path = [Point((pt[0], pt[1])) for pt in budgeted_path]
    # else:
    #   shapely_path = [pt.shapelyPoint() for pt in budgeted_path]

    ls = LineString(shapely_path)

    distances = np.array([ls.distance(pt) for pt in self.pts])

    distances = distances.reshape(self.info_field.shape)

    if self.method == 'binary':
      mask = distances <= self.radius

    elif self.method == 'linear':
      mask = (self.radius - np.minimum(distances, self.radius)) / self.radius

    elif self.method == 'squared':
      mask = ((self.radius - np.minimum(distances, self.radius)) / self.radius)**2

    return mask

  def getExactScore(self, path, budget=float('inf'), plot=False):
    budgeted_path, budgeted_path_len = applyBudget(path, budget)
    mask = np.zeros(self.info_field.shape)

    if len(budgeted_path) < 2:
      return 0.0, 0.0

    if isinstance(budgeted_path, np.ndarray):
      #shapely_path = [Location(xlon=pt[0], ylat=pt[1]).shapelyPoint() for pt in budgeted_path]
      shapely_path = [Point((pt[0], pt[1])) for pt in budgeted_path]
    # else:
    #   shapely_path = [pt.shapelyPoint() for pt in budgeted_path]

    ls = LineString(shapely_path)

    distances = np.array([ls.distance(pt) for pt in self.pts])

    distances = distances.reshape(self.info_field.shape)

    if self.method == 'binary':
      mask = distances <= self.radius

    elif self.method == 'linear':
      mask = (self.radius - np.minimum(distances, self.radius)) / self.radius

    elif self.method == 'squared':
      mask = ((self.radius - np.minimum(distances, self.radius)) / self.radius)**2

    if plot:
      plt.figure()
      ax = plt.gca()
      plt.pcolor(self.x_ticks, self.y_ticks, (mask*self.info_field).transpose())
      plt.show(False)

    return np.sum(mask*self.info_field), 0.0


  def getMask(self, path, budget=float('inf')):
    distances = np.ones(self.info_field.shape)*float('inf')
    budgeted_path, budgeted_path_len = applyBudget(path, budget)

    if len(budgeted_path) < 2:
      return 0., 0.

    if isinstance(budgeted_path, np.ndarray):
      query_pts = np.zeros((0,2))

      for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
        if not np.array_equal(p1, p2):
          unit_vec = (p2 - p1) / np.linalg.norm(p2-p1)

          dx_step, dy_step = self.step_size*unit_vec


          if dx_step != 0:
            xs = np.arange(p1[0], p2[0], dx_step)
          if dy_step != 0:
            ys = np.arange(p1[1], p2[1], dy_step)
          if dx_step == 0:
            xs = np.ones(ys.shape)*p1[0]
          if dy_step == 0:
            ys = np.ones(xs.shape)*p1[1]

          xs = xs[:len(ys)] # Ensure the two are of equal length
          ys = ys[:len(xs)]

          segment_query_pts = np.vstack((xs, ys)).transpose()

          query_pts = np.vstack((query_pts, segment_query_pts))

      query_pts = np.vstack((query_pts, p2))


    else:
      # query_pts = []
      # for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
      #   query_pts += locationArange(p1, p2, self.step_size)
      # query_pts = [pt.asTuple() for pt in query_pts]
      # query_pts.append(p2.asTuple())
      print('functionality not defined, needs an ndarray of points')

    for pt in query_pts:
      z = np.sqrt((self.xx-pt[0])**2 + (self.yy-pt[1])**2)
      distances = np.minimum(z, distances)


    if self.method == 'binary':
      mask = distances <= self.radius

    elif self.method == 'linear':
      mask = (self.radius - np.minimum(distances, self.radius)) / self.radius

    elif self.method == 'squared':
      mask = ((self.radius - np.minimum(distances, self.radius)) / self.radius)**2

    return mask

  def getScore(self, path, budget=float('inf')):
    distances = np.ones(self.info_field.shape)*float('inf')
    budgeted_path, budgeted_path_len = applyBudget(path, budget)

    if len(budgeted_path) < 2:
      return 0., 0.

    if isinstance(budgeted_path, np.ndarray):
      query_pts = np.zeros((0,2))

      for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
        unit_vec = (p2 - p1) / np.linalg.norm(p2-p1)

        dx_step, dy_step = self.step_size*unit_vec


        if dx_step != 0:
          xs = np.arange(p1[0], p2[0], dx_step)
        if dy_step != 0:
          ys = np.arange(p1[1], p2[1], dy_step)
        if dx_step == 0:
          xs = np.ones(ys.shape)*p1[0]
        if dy_step == 0:
          ys = np.ones(xs.shape)*p1[1]

        xs = xs[:len(ys)] # Ensure the two are of equal length
        ys = ys[:len(xs)]

        segment_query_pts = np.vstack((xs, ys)).transpose()

        query_pts = np.vstack((query_pts, segment_query_pts))

      query_pts = np.vstack((query_pts, p2))


    else:
      print('functionality not defined, needs an ndarray of points')
      # query_pts = []
      # for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
      #   query_pts += locationArange(p1, p2, self.step_size)
      # query_pts = [pt.asTuple() for pt in query_pts]
      # query_pts.append(p2.asTuple())

    for pt in query_pts:
      z = np.sqrt((self.xx-pt[0])**2 + (self.yy-pt[1])**2)
      distances = np.minimum(z, distances)


    if self.method == 'binary':
      mask = distances <= self.radius

    elif self.method == 'linear':
      mask = (self.radius - np.minimum(distances, self.radius)) / self.radius

    elif self.method == 'squared':
      mask = ((self.radius - np.minimum(distances, self.radius)) / self.radius)**2

    # if plot:
    #   plt.figure()
    #   ax = plt.gca()
    #   plt.pcolor(self.x_ticks, self.y_ticks, (mask*self.info_field).transpose())
    #   plt.show(False)

    return np.sum(mask*self.info_field), 1.0

  # def getHomotopyScore(self, query_path):
  #
  #   query_h_sig = HSignature.fromPath(query_path, self.rep_pts)
  #
  #   if query_h_sig.contains(self.reference_h_sig) or self.reference_h_sig.contains(query_h_sig):
  #     return 0.
  #   else:
  #     return -1.


class PathEvaluatorAlongPath(PathEvaluator):
  """docstring for PathEvaluatorAlongPath"""
  def __init__(self, information_interp, cost_interp, yaml_eval, rep_pts=[]):
    self.infoInterpFn = information_interp
    self.costInterpFn = cost_interp

    self.step_size = yaml_eval['step_size']
    self.rep_pts = rep_pts

    #self.reference_h_sig = HSignature(())


  def getScore(self, path, budget=float('inf')):
    budgeted_path, budgeted_path_len = applyBudget(path, budget)

    if len(budgeted_path) < 2:
      return 0., 0.

    if isinstance(budgeted_path, np.ndarray):
      query_pts = np.zeros((0,2))

      for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
        if not np.array_equal(p1, p2):
          unit_vec = (p2 - p1) / np.linalg.norm(p2-p1)

          dx_step, dy_step = self.step_size*unit_vec


          if dx_step != 0:
            xs = np.arange(p1[0], p2[0], dx_step)
          if dy_step != 0:
            ys = np.arange(p1[1], p2[1], dy_step)
          if dx_step == 0:
            xs = np.ones(ys.shape)*p1[0]
          if dy_step == 0:
            ys = np.ones(xs.shape)*p1[1]

          xs = xs[:len(ys)] # Ensure the two are of equal length
          ys = ys[:len(xs)]

          segment_query_pts = np.vstack((xs, ys)).transpose()

          query_pts = np.vstack((query_pts, segment_query_pts))

      query_pts = np.vstack((query_pts, p2))


    else:
      # query_pts = []
      # for p1, p2 in zip(budgeted_path[:-1], budgeted_path[1:]):
      #   query_pts += locationArange(p1, p2, self.step_size)
      print('functionality not defined, needs an ndarray of points')

      # query_pts = [pt.asTuple() for pt in query_pts]
      # query_pts.append(p2.asTuple())

    info_scores = self.infoInterpFn(query_pts)
    # cost_scores = self.costInterpFn(query_pts)

    np.nan_to_num(info_scores, copy=False)
    return (np.nanmean(info_scores) / self.step_size) * budgeted_path_len, 0.0 # (np.mean(cost_scores) / self.step_size) * budgeted_path_len

  # def getHomotopyScore(self, query_path):
  #
  #   query_h_sig = HSignature.fromPath(query_path, self.rep_pts)
  #
  #   if query_h_sig.contains(self.reference_h_sig) or self.reference_h_sig.contains(query_h_sig):
  #     return 0.
  #   else:
  #     return -1.


def applyBudget(path, budget, verbose=False):
  distances = [np.linalg.norm(path[pt_idx-1] - path[pt_idx]) if pt_idx > 0 else 0 for pt_idx in range(len(path))]
  cumulative_distances = np.cumsum(distances)
  budgeted_path = [path[pt_idx] for pt_idx in range(len(path)) if cumulative_distances[pt_idx] <= budget]

  if len(budgeted_path) < len(path):
    budget_diff = budget - cumulative_distances[len(budgeted_path)-1]
    assert budget_diff > 0
    unit_vec = (path[len(budgeted_path)] - budgeted_path[-1]) / np.linalg.norm(budgeted_path[-1] - path[len(budgeted_path)])
    budgeted_path.append(budgeted_path[-1] + unit_vec*budget_diff)

  else:
    budgeted_path_len = cumulative_distances[-1]

  if isinstance(path, np.ndarray):
    budgeted_path = np.vstack(budgeted_path)

  if verbose:
    print("Initial:", cumulative_distances[-1])
    print("Final:", budgeted_cumulative_distances[-1])

  return budgeted_path, min(budget, cumulative_distances[-1])




# if __name__ == '__main__':
#   n_ticks = 51
#
#   x_ticks = np.linspace(-25., 25., n_ticks)
#   y_ticks = np.linspace(-25., 25., n_ticks)
#
#   info_field = np.random.random((len(x_ticks), len(y_ticks)))
#   cost_field = np.random.random((len(x_ticks), len(y_ticks)))
#
#   infoInterpFn = RegularGridInterpolator([x_ticks, y_ticks], info_field)
#   costInterpFn = RegularGridInterpolator([x_ticks, y_ticks], cost_field)
#
#
#
#
#   path = [Location(xlon=-10.,ylat=-10.),
#           Location(xlon=10.,ylat=-10.),
#           Location(xlon=0., ylat=0.0)]
#
#   rep_pts = []
#
#   yaml_eval = {'radius':20, 'method':'linear'}
#   evaluator_radius = PathEvaluatorWithRadius(info_field, cost_field, x_ticks, y_ticks, yaml_eval)
#
#   yaml_eval = {'step_size':.2}
#   evaluator_linear = PathEvaluator(infoInterpFn, infoInterpFn, yaml_eval)
#
#   paths = np.random.random((1000, 5 ,2))*50 - 25
#
#
#   exact_scores = []
#   tick = time.time()
#   for path in paths:
#     score, _ = evaluator_radius.getExactScore(path, budget=100)
#     exact_scores.append(score)
#   print("With Exact Distance: %.3f, %.3f" % (time.time() - tick, np.mean(exact_scores)))
#
#   appx_scores = []
#   tick = time.time()
#   for path in paths:
#     score, _ = evaluator_radius.getScore(path, budget=100, plot=True)
#     appx_scores.append(score)
#   print("With Appx Distance: %.3f, %.3f" % (time.time() - tick, np.mean(appx_scores)))
#
#   linear_scores = []
#   tick = time.time()
#   for path in paths:
#     score, _ = evaluator_linear.getScore(path, budget=100)
#     linear_scores.append(score)
#   print("Linear: %.3f, %.3f" % (time.time() - tick, np.mean(linear_scores)))
