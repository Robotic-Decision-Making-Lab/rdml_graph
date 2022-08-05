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
# EnsembleLearning.py
# Written Ian Rankin - April 2022
#
# Functions to learn various ensembles of decision tree's
# In particular Random Forests.


from rdml_graph.core import TreeNode
from rdml_graph.decision_tree.DecisionTreeHelper import \
        classification_importance, regression_importance, \
        class_plurality, reg_plurality, \
        same_class, bin_category_split, bin_float_split, default_attribute_func
from rdml_graph.decision_tree.DecisionLearner import learn_decision_tree

import random
import math
import numpy as np

from multiprocessing import Pool
import time

import pdb

## learn_decision_tree from the data in parameter X
# @param X - the input data
# REQUIRED
# @param importance_func - the importance of different attributes
# @param attribute_func - the function to find and generate the next best node
#       of the decision tree must be
#       type - func(X, importance_func, types, parent)
#
# OPTIONAL
# @param plurality_func - [opt] return the leaf node for a particular part of the tree
# @param same_func - checks if the data input is all the same and spliting
#               can stop.
# @param types - [opt] a list of types of the input tree to use with the attribute function
# @param with_labels - [opt] defines if it is a classification or regression problem with labels on X
# @param max_depth - [opt] the max_depth to allow the tree to go
#
# RETURN
# @return ensemble learning object
def learn_random_forest(X, \
        types, \
        num_trees, \
        importance_func, \
        attribute_func= default_attribute_func, \
        max_depth=float('inf'),\
        plurality_func=class_plurality, \
        same_func=same_class, \
        with_labels=True, \
        num_threads = 12, \
        progress_cb = None):
    # Start of  function

    # create sub-samples of dataset

    size_of_subsets = int(math.ceil(len(X)/2))

    subsets = [random.sample(X, size_of_subsets) for i in range(num_trees)]

    iteratable = [(subset, types, importance_func, attribute_func, \
                    max_depth, plurality_func, same_func, with_labels) for subset in subsets]

    # t1 = time.time()
    # This does the same but without any known progress.
    # with Pool(num_threads) as p:
    #     trees = p.starmap(learn_decision_tree, iteratable)
    # t2 = time.time()

    with Pool(num_threads) as p:
        #trees = []
        # start running
        results = [p.apply_async(learn_decision_tree, it) for it in iteratable]

        # check results
        completed = 0
        running = 0
        while completed != len(results):
            completed = 0
            running = 0
            for i in range(len(results)):
                try:
                    if results[i].successful():
                        completed += 1
                    else:
                        print('tree i has an error: ' + str(i))
                except ValueError:
                    running += 1

            progress = (completed / len(results))
            #print('Progress: ' + str(progress))
            time.sleep(0.2)
            if progress_cb is not None:
                progress_cb(progress)
        # end while loop
        trees = [result.get() for result in results]
    # t3 = time.time()
    # print('Total time without progress: ' + str(t2 - t1))
    # print('Total time with progress: ' + str(t3 - t2))


    trees = [t[0] for t in trees]

    model = Ensemble(trees, samples=X)

    return model, 0


def predict_parallel(input, tree):
    return tree.traverse(input)


class Ensemble:

    ## init
    # constructor for the Ensemble learning algorithm
    # @param trees a list of trees and id's
    def __init__(self, trees, weights = None, samples = None):
        self.trees = trees

        if weights is None:
            self.weights = np.ones(len(trees)) / len(trees)
        else:
            self.weights = weights

        self.samples = samples

    ## traverses the tree to get to the leaf node.
    def traverse(self, input, num_threads = 12):
        iteratable = [(input, tree) for tree in self.trees]

        with Pool(num_threads) as p:
            y_pred = p.starmap(predict_parallel, iteratable)

        predictions = np.array(y_pred)

        output_prediction = np.dot(predictions, self.weights)
        return output_prediction































#
