# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""
Various utilities.
"""

import tensorflow as tf
import os
import sys
import copy
import numpy as np
import math
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn

import pdb

class ArraySaver(object):
    """A simple class helping with saving/loading numpy arrays from files.

    This class allows to save / load numpy arrays, while storing them either
    on disk or in memory.
    """

    def __init__(self, mode='ram', workdir=None):
        self._mode = mode
        self._workdir = workdir
        self._global_arrays = {}

    def save(self, name, array):
        if self._mode == 'ram':
            self._global_arrays[name] = copy.deepcopy(array)
        elif self._mode == 'disk':
            create_dir(self._workdir)
            np.save(o_gfile((self._workdir, name), 'wb'), array)
        else:
            assert False, 'Unknown save / load mode'

    def load(self, name):
        if self._mode == 'ram':
            return self._global_arrays[name]
        elif self._mode == 'disk':
            return np.load(o_gfile((self._workdir, name), 'rb'))
        else:
            assert False, 'Unknown save / load mode'

def create_dir(d):
    if not tf.gfile.IsDirectory(d):
        tf.gfile.MakeDirs(d)

class File(tf.gfile.GFile):
    """Wrapper on GFile extending seek, to support what python file supports."""
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)

def o_gfile(filename, mode):
    """Wrapper around file open, using gfile underneath.

    filename can be a string or a tuple/list, in which case the components are
    joined to form a full path.
    """
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)

def listdir(dirname):
    return tf.gfile.ListDirectory(dirname)

def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)

def discretizer(target, num_bins=20):
    """Discretize target based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
                                target[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(zs, ys):
  """Compute discrete mutual information."""
  num_codes = zs.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], zs[i, :])
  return m

def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h

def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

def compute_score_matrix(mus, ys, mus_test, ys_test):
    """Compute score matrix as described in Kumar et al."""
    mus = np.transpose(mus)
    mus_test = np.transpose(mus_test)
    ys = np.transpose(ys)
    ys_test = np.transpose(ys_test)
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            mu_i_test = mus_test[i, :]
            y_j_test = ys_test[j, :]
            classifier = sklearn.svm.LinearSVC(C=0.01, class_weight="balanced")
            classifier.fit(mu_i[:, np.newaxis], y_j)
            pred = classifier.predict(mu_i_test[:, np.newaxis])
            score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix

def sample_factors(opts, batch_size, data):
    # sampling factors
    factor_sizes = data.factor_sizes
    factor_num = len(factor_sizes)
    # sampling batch of factors
    factors = np.zeros((batch_size,factor_num))
    for i in range(factor_num):
        factors[:,i] = np.random.randint(factor_sizes[i],size=batch_size)
    return factors

def sample_images(opts, dataset, data, factors):
    # generating images from factors
    images, labels = data.sample_observations_from_factors(dataset, factors)
    # indices = np.dot(factors, data.factor_bases).astype(dtype=np.int32)
    # images, labels = sample_from_factor_indices(data, indices)
    return images
