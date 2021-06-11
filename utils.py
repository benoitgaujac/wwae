# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""
Various utilities.
"""

import pdb
import sklearn
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import os
import sys
import copy
import numpy as np
import math
import logging
import matplotlib
matplotlib.use("Agg")

import typing


class ArraySaver(object):
    """A simple class helping with saving/loading numpy arrays from files.

    This class allows to save / load numpy arrays, while storing them either
    on disk or in memory.
    """

    def __init__(self, mode='ram', workdir=None) -> None:
        self._mode: str = mode
        self._workdir: typing.Optional[str] = workdir
        self._global_arrays: dict = {}

    def save(self, name: str, array) -> None:
        if self._mode == 'ram':
            self._global_arrays[name] = copy.deepcopy(array)
        elif self._mode == 'disk':
            create_dir(self._workdir)
            np.save(o_gfile((self._workdir, name), 'wb'), array)
        else:
            assert False, 'Unknown save / load mode'

    def load(self, name: str) -> object:
        if self._mode == 'ram':
            return self._global_arrays[name]
        elif self._mode == 'disk':
            return np.load(o_gfile((self._workdir, name), 'rb'))
        else:
            assert False, 'Unknown save / load mode'


def create_dir(d: str) -> None:
    if not tf.io.gfile.isdir(d):
        tf.io.gfile.mkdir(d)


class File(tf.io.gfile.GFile):
    """Wrapper on GFile extending seek, to support what python file supports."""

    def __init__(self, *args) -> None:
        super(File, self).__init__(*args)

    def seek(self, position: int, whence=0) -> None:
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)


def o_gfile(filename, mode: str) -> File:
    """Wrapper around file open, using gfile underneath.

    filename can be a string or a tuple/list, in which case the components are
    joined to form a full path.
    """
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)


def listdir(dirname: str) -> object:
    return tf.io.gfile.ListDirectory(dirname)


def get_batch_size(inputs: tf.Tensor) -> typing.Any:
    return tf.cast(tf.shape(inputs)[0], tf.float32)
