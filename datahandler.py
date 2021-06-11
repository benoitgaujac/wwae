# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import shutil
import random
import logging
import gzip
import zipfile
import tensorflow.compat.v1 as tf
import numpy as np
from six.moves import cPickle
import urllib.request
import requests
from scipy.io import loadmat
from sklearn.feature_extraction import image
import struct
from PIL import Image
import sys
import tarfile
import h5py
from math import ceil

import utils

import pdb
import typing

# Path to data
# data_dir = '../data'

datashapes = {}
datashapes['gmm'] = [12, 12, 1]
datashapes['mnist'] = [32, 32, 1]
datashapes['shifted_mnist'] = [64, 64, 1]
datashapes['shifted_3pos_mnist'] = [64, 64, 1]
datashapes['rotated_mnist'] = [32, 32, 1]
datashapes['svhn'] = [32, 32, 3]
datashapes['cifar10'] = [32, 32, 3]
datashapes['celebA'] = [64, 64, 3]


def _data_dir(opts: dict) -> str:
    _data_dir = os.path.join(opts['data_dir'], opts['dataset'])
    if opts['dataset']=='mnist':
        data_path = _data_dir
    elif opts['dataset']=='shifted_mnist':
        data_path = os.path.join(opts['data_dir'],'mnist')
    elif opts['dataset']=='shifted_3pos_mnist':
        data_path = os.path.join(opts['data_dir'],'mnist')
    elif opts['dataset']=='rotated_mnist':
        data_path = os.path.join(opts['data_dir'],'mnist')
    elif opts['dataset']=='svhn':
        data_path = _data_dir
    elif opts['dataset']=='cifar10':
        data_path = os.path.join(_data_dir, 'cifar-10-batches-py')
    elif opts['dataset']=='celebA':
        data_path = os.path.join(_data_dir,'img_align_celeba')
    else:
        assert False, 'Unknow {} dataset'.format(opts['dataset'])
    assert os.path.isdir(data_path), '{} dir. doesnt exist'. format(opts['dataset'])
    return data_path

def _load_cifar_batch(fpath: str) -> tf.Tensor:
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.

    # Returns
        A ndarray data.
    """
    f = utils.o_gfile(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']

    data = data.reshape(data.shape[0], 3, 32, 32).astype(dtype=np.float32)
    return data

def _shift_mnist_tf(x: tf.Tensor) -> tf.Tensor:
    # padding mnist img
    # paddings = [[2,2], [2,2], [0,0]]
    # x_pad = tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
    # shape = x_pad.shape.as_list()
    shape = x.shape.as_list()
    transformed_shape = datashapes['shifted_mnist']
    img = tf.zeros(transformed_shape,tf.float32)
    # sample cluster pos
    i = tf.random.uniform([], 0, 2, tf.int32)
    # pos_x = i*int(3*shape[0]/4)
    # pos_y = i*int(3*shape[1]/4)
    pos_x = i*int(3*transformed_shape[0]/4-shape[0])
    pos_y = i*int(3*transformed_shape[1]/4-shape[1])
    # sample shift
    # shift_x = tf.random.uniform([], 0, int(shape[0]/4)+1, tf.int32)
    # shift_y = tf.random.uniform([], 0, int(shape[1]/4)+1, tf.int32)
    shift_x = tf.random.uniform([], 0, int(transformed_shape[0]/4)+1, tf.int32)
    shift_y = tf.random.uniform([], 0, int(transformed_shape[1]/4)+1, tf.int32)
    # create img
    # paddings = [[pos_x+shift_x, shape[0]-pos_x-shift_x],
    #             [pos_y+shift_y, shape[1]-pos_y-shift_y],
    #             [tf.zeros([],tf.int32), tf.zeros([],tf.int32)]]
    paddings = [[pos_x+shift_x, transformed_shape[0]-shape[0] - (pos_x+shift_x)],
                [pos_y+shift_y, transformed_shape[0]-shape[0] - (pos_y+shift_y)],
                [tf.zeros([],tf.int32), tf.zeros([],tf.int32)]]
    paddings = tf.stack([tf.stack(t,0) for t in paddings], 0)
    # img = tf.pad(x_pad, paddings, mode='CONSTANT', constant_values=0.)
    img = tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)

    return tf.reshape(img, datashapes['shifted_mnist'])

def _shift_mnist_3pos_tf(x: tf.Tensor) -> tf.Tensor:
    # padding mnist img
#    paddings = [[2,2], [2,2], [0,0]]
#    x_pad = tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
    shape = x.shape.as_list()
    #shape = x.shape.as_list()
    transformed_shape = datashapes['shifted_mnist']
    img = tf.zeros(transformed_shape,tf.float32)
    # sample cluster pos
    i = tf.random.uniform([], 0, 3, tf.int32)
    pos_x = 8*(i + 1)
    pos_y = pos_x

    paddings = [[pos_x, transformed_shape[0]-shape[0] - (pos_x)],
                [pos_y, transformed_shape[0]-shape[0] - (pos_y)],
                [tf.zeros([],tf.int32), tf.zeros([],tf.int32)]]
    paddings = tf.stack([tf.stack(t,0) for t in paddings], 0)
    # img = tf.pad(x_pad, paddings, mode='CONSTANT', constant_values=0.)
    img = tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)

    return tf.reshape(img, datashapes['shifted_mnist'])

def _shift_mnist_np(x: np.ndarray) -> np.ndarray:
    # padding mnist img
    # paddings = [[2,2], [2,2], [0,0]]
    # x_pad = np.pad(x, paddings, mode='constant', constant_values=0.)
    # shape = x_pad.shape
    shape = x.shape
    transformed_shape = datashapes['shifted_mnist']
    # create img
    img = np.zeros(transformed_shape)
    # sample cluster pos
    i = np.random.binomial(1, 0.5)
    # pos_x = i*int(3*shape[0]/4)
    # pos_y = i*int(3*shape[1]/4)
    pos_x = i*int(3*transformed_shape[0]/4-shape[0])
    pos_y = i*int(3*transformed_shape[1]/4-shape[1])
    # sample shift
    # shift_x = np.random.randint(0, int(shape[0]/4)+1)
    # shift_y = np.random.randint(0, int(shape[1]/4)+1)
    shift_x = np.random.randint(0, int(transformed_shape[0]/4)+1)
    shift_y = np.random.randint(0, int(transformed_shape[1]/4)+1)
    # place digit
    # img[pos_x+shift_x:shape[0]+pos_x+shift_x, pos_y+shift_y:shape[1]+pos_y+shift_y] = x_pad
    img[pos_x+shift_x:shape[0]+pos_x+shift_x, pos_y+shift_y:shape[1]+pos_y+shift_y] = x

    return img

def _shift_mnist_3pos_np(x: np.ndarray) -> np.ndarray:
    # padding mnist img
    # paddings = [[2,2], [2,2], [0,0]]
    # x_pad = np.pad(x, paddings, mode='constant', constant_values=0.)
    # shape = x_pad.shape
    shape = x.shape
    transformed_shape = datashapes['shifted_mnist']
    # create img
    img = np.zeros(transformed_shape)
    # sample cluster pos
    i = np.random.randint(3)
#    pos_x = i*int(transformed_shape[0]/4)
#    pos_y = i*int(transformed_shape[1]/4)
    if int(i)==0:
        pos_x = 8; pos_y = 8
    elif int(i)==1:
        pos_x = 16; pos_y = 16
    else:
        pos_x = 24; pos_y = 24
    # place digit
    # img[pos_x+shift_x:shape[0]+pos_x+shift_x, pos_y+shift_y:shape[1]+pos_y+shift_y] = x_pad
    img[pos_x:shape[0]+pos_x, pos_y:shape[1]+pos_y] = x

    return img

def _rotate_mnist_tf(x: tf.Tensor) -> tf.Tensor:
    #padding mnist img
    paddings = [[2,2], [2,2], [0,0]]
    x_pad = tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
    # rot image with 0.5 prob
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    img = tf.cond(choice < 0.5, lambda: x_pad, lambda: tf.image.rot90(x_pad))

    return img

def _rotate_mnist_np(x: np.ndarray) -> np.ndarray:
    # padding mnist img
    paddings = [[2,2], [2,2], [0,0]]
    x_pad = np.pad(x, paddings, mode='constant', constant_values=0.)
    # rot image with 0.5 prob
    choice = np.random.randint(0,2)
    img = np.where(choice==0, x_pad, np.rot90(x_pad))

    return img

def _gmm_generator(dataset_size: int) -> typing.Generator[np.ndarray, None, None]:

    logits_shape = [int(datashapes['gmm'][0]/2),int(datashapes['gmm'][1]/2),datashapes['gmm'][2]]
    zeros = np.zeros(logits_shape)
    # generate data
    n = 0
    while n<dataset_size:
        # choose mixture
        mu = zeros
        choice = np.random.randint(0,2)
        mu[3*choice:3*choice+3,3*choice:6*choice+3] = np.ones((3,3,1))
        mu[1+3*choice,1+3*choice] = [1.5]
        # sample cat. logits
        logits = np.random.normal(mu,.1,size=logits_shape).reshape((-1))
        p = np.exp(logits) / np.sum(np.exp(logits))
        a = np.arange(np.prod(logits_shape))
        # sample pixel idx
        idx = np.random.choice(a,size=1,p=p)[0]
        i = int(idx / 6.)
        j = idx % 6
        # generate obs
        x = np.zeros(datashapes['gmm'])
        x[2*i:2*i+2,2*i:2*i+2] = np.ones((2,2,1))
        yield x
        n+=1

def _sample_gmm(batch_size: int) -> np.ndarray:

    obs = np.zeros([batch_size,]+datashapes['gmm'])
    logits_shape = [int(datashapes['gmm'][0]/2),int(datashapes['gmm'][1]/2),datashapes['gmm'][2]]
    for n in range(batch_size):
        # choose mixture
        mu = np.zeros(logits_shape)
        choice = np.random.randint(0,2)
        mu[3*choice:3*choice+3,3*choice:6*choice+3] = np.ones((3,3,1))
        mu[1+3*choice,1+3*choice] = [1.5]
        # sample cat. logits
        logits = np.random.normal(mu,.1,size=logits_shape).reshape((-1))
        p = np.exp(logits) / np.sum(np.exp(logits))
        a = np.arange(np.prod(logits_shape))
        # sample pixel idx
        idx = np.random.choice(a,size=1,p=p)[0]
        i = int(idx / 6.)
        j = idx % 6
        # generate obs
        x = np.zeros(datashapes['gmm'])
        x[2*i:2*i+2,2*i:2*i+2] = np.ones((2,2,1))
        obs[n] = x

    return obs


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts: dict) -> None:
        self.dataset: str = opts['dataset']
        self.crop_style: str = opts['celebA_crop']
        # load data
        logging.error('\n Loading {}.'.format(self.dataset))
        self._init_dataset(opts)
        logging.error('Loading Done.')

    def _init_dataset(self, opts: dict) -> None:
        """Load a dataset and fill all the necessary variables.

        """
        if self.dataset == 'gmm':
            self._load_gmm(opts)
        elif self.dataset == 'mnist':
            self._load_mnist(opts)
        elif self.dataset == 'shifted_mnist':
            self._load_shift_mnist(opts)
        elif self.dataset == 'shifted_3pos_mnist':
            self._load_shift_3pos_mnist(opts)
        elif self.dataset == 'rotated_mnist':
            self._load_rot_mnist(opts)
        elif self.dataset == 'svhn':
            self._load_svhn(opts)
        elif self.dataset == 'cifar10':
            self._load_cifar10(opts)
        elif self.dataset == 'celebA':
            self._load_celebA(opts)
        else:
            raise ValueError('Unknown %s dataset' % self.dataset)

    def _load_gmm(self, opts: dict) -> None:
        """Create GMM dataset.

        """
        # plot set
        self.data_plot = _sample_gmm(opts['evaluate_num_pics'])
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        self.data_vizu = _sample_gmm(opts['plot_num_pics'])
        # dataset size
        self.train_size = 10000
        self.test_size = 5000
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_generator(_gmm_generator,
                                output_types=tf.float32,
                                output_shapes =self.data_shape,
                                args=[self.train_size])
        dataset_test = tf.data.Dataset.from_generator(_gmm_generator,
                                output_types=tf.float32,
                                output_shapes = self.data_shape,
                                args=[self.test_size])
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        # dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        # dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()

    def _load_mnist(self, opts: dict) -> None:
        """Load data from MNIST or ZALANDO files.

        """
        self.data_dir = _data_dir(opts)
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(num_data,size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # pad data to 32x32
        def pad_mnist(x):
            paddings = [[2,2], [2,2], [0,0]]
            return tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
        dataset_train = dataset_train.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()

    def _load_shift_mnist(self, opts: dict) -> None:
        """Load 0s and 1s digits from MNIST and
        shift randomly digit in top-left or bottom-right corner

        """
        self.data_dir = _data_dir(opts)
        # loading label
        tr_Y, te_Y = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000*1), dtype=np.uint8)
            tr_Y = loaded.reshape((60000,)).astype(np.int64)
        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000*1), dtype=np.uint8)
            te_Y = loaded.reshape((10000,)).astype(np.int64)
        Y = np.concatenate((tr_Y, te_Y), axis=0)
        zeros_idx = np.where(Y==0, 1, 0)
        ones_idx = np.where(Y==1, 1, 0)
        zeros_ones_idx = zeros_idx + ones_idx
        Y = Y[zeros_ones_idx==1]
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        X = X[zeros_ones_idx==1]
        self.all_data = X / 255.
        self.all_labels = Y
        num_data = len(Y)
        # plot set
        idx = np.random.randint(num_data,size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-1000:
            tr_stop = num_data - 1000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-1000:]]
        # dataset size
        self.train_size = len(data_train)
        self.test_size = len(data_test)
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # transform mnist
        dataset_train = dataset_train.map(_shift_mnist_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(_shift_mnist_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()

    def _load_shift_3pos_mnist(self, opts: dict) -> None:
        """Load 1s digits from MNIST and
        shift randomly digit in top-left, middle, or bottom-right corner
        """
        self.data_dir = _data_dir(opts)
        # loading label
        tr_Y, te_Y = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000*1), dtype=np.uint8)
            tr_Y = loaded.reshape((60000,)).astype(np.int64)
        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000*1), dtype=np.uint8)
            te_Y = loaded.reshape((10000,)).astype(np.int64)
        Y = np.concatenate((tr_Y, te_Y), axis=0)
        ones_idx = np.where(Y==1, 1, 0)
        Y = Y[ones_idx==1]
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        X = X[ones_idx==1]
        self.all_data = X / 255.
        self.all_labels = Y
        num_data = len(Y)
        # plot set
        idx = np.random.randint(num_data,size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-1000:
            tr_stop = num_data - 1000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-1000:]]
        # dataset size
        self.train_size = len(data_train)
        self.test_size = len(data_test)
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # transform mnist
        dataset_train = dataset_train.map(_shift_mnist_3pos_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(_shift_mnist_3pos_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()



    def _load_rot_mnist(self, opts: dict) -> None:
        """Load 1s and 5s digits from MNIST and
        shift randomly digit in top-left or bottom-right corner

        """
        self.data_dir = _data_dir(opts)
        # loading label
        tr_Y, te_Y = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000*1), dtype=np.uint8)
            tr_Y = loaded.reshape((60000,)).astype(np.int64)
        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000*1), dtype=np.uint8)
            te_Y = loaded.reshape((10000,)).astype(np.int64)
        Y = np.concatenate((tr_Y, te_Y), axis=0)
        ones_idx = np.where(Y==1, 1, 0)
        three_idx = np.where(Y==3, 1, 0)
        seven_idx = np.where(Y==7, 1, 0)
        eight_idx = np.where(Y==8, 1, 0)
        all_idx = ones_idx + three_idx + seven_idx + eight_idx
        Y = Y[all_idx==1]
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        X = X[all_idx==1]
        self.all_data = X / 255.
        self.all_labels = Y
        num_data = len(Y)
        # plot set
        idx = np.random.randint(num_data,size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-1000:
            tr_stop = num_data - 1000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-1000:]]
        # dataset size
        self.train_size = len(data_train)
        self.test_size = len(data_test)
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # transform mnist
        dataset_train = dataset_train.map(_rotate_mnist_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(_rotate_mnist_tf,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()

    def _load_svhn(self, opts: dict) -> None:
        """Load data from SVHN files.

        """
        # Helpers to process raw data
        def convert_imgs_to_array(img_array):
            rows = datashapes['svhn'][0]
            cols = datashapes['svhn'][1]
            chans = datashapes['svhn'][2]
            num_imgs = img_array.shape[3]
            # Note: not the most efficent way but can monitor what is happening
            new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
            for x in range(0, num_imgs):
                # TODO reuse normalize_img here
                chans = img_array[:, :, :, x]
                # # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
                # norm_vec = (255-chans)*1.0/255.0
                new_array[x] = chans
            return new_array

        self.data_dir = _data_dir(opts)
        # loading images
        # Training data
        file_path = os.path.join(self.data_dir,'train_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        tr_X = convert_imgs_to_array(imgs)
        file.close()
        if opts['use_extra']:
            file_path = os.path.join(self.data_dir,'extra_32x32.mat')
            file = open(file_path, 'rb')
            data = loadmat(file)
            imgs = data['X']
            extra_X = convert_imgs_to_array(imgs)
            extra_X = extra_X
            file.close()
            # concatenate training and extra
            tr_X = np.concatenate((tr_X,extra_X), axis=0)
        # Testing data
        file_path = os.path.join(self.data_dir,'test_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        te_X = convert_imgs_to_array(imgs)
        file.close()
        X = np.concatenate((tr_X, te_X), axis=0)
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_cifar10(self, opts: dict) -> None:
        """Load data from MNIST or ZALANDO files.

        """

        self.data_dir = _data_dir(opts)
        # loading data
        num_train_samples = 50000
        data_dir = _data_dir(opts)
        x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(self.data_dir, 'data_batch_' + str(i))
            data = _load_cifar_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        x_train = x_train.transpose(0, 2, 3, 1)
        fpath = os.path.join(self.data_dir, 'test_batch')
        x_test = _load_cifar_batch(fpath)
        x_test = x_test.transpose(0, 2, 3, 1)
        X = np.vstack([x_train, x_test])
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_celebA(self, opts: dict) -> None:
        """Load CelebA
        """
        num_data = 202599
        self.data_dir = _data_dir(opts)
        self.all_data = np.array([os.path.join(self.data_dir,'%.6d.jpg') % i for i in range(1, num_data + 1)])
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # map files paths to image with tf.io.decode_jpeg
        def process_path(file_path):
            # reading .jpg file
            image_file = tf.read_file(file_path)
            im_decoded = tf.cast(tf.image.decode_jpeg(image_file, channels=3), dtype=tf.dtypes.float32)
            # crop and resize
            width = 178
            height = 218
            new_width = 140
            new_height = 140
            if self.crop_style == 'closecrop':
                # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
                im = tf.image.crop_and_resize(tf.expand_dims(im_decoded,axis=0),
                                        np.array([[top / (height-1), right / (width-1), bottom / (height-1), left / (width-1)]]),
                                        [0,],
                                        (64,64),
                                        method='bilinear', extrapolation_value=0)
            # elif self.crop_style == 'resizecrop':
            #     # This method was used in ALI, AGE, ...
            #     im = im.resize((64, 78), Image.ANTIALIAS)
            #     im = im.crop((0, 7, 64, 64 + 7))
            else:
                assert False, '{} not implemented.'.format(self.crop_style)
            return tf.reshape(im, datashapes['celebA']) / 255.
        dataset_train = dataset_train.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def init_iterator(self, sess: tf.Session) -> typing.Tuple[typing.Any, typing.Any]:
        sess.run([self.iterator_train.initializer,self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run([self.iterator_train.string_handle(),self.iterator_test.string_handle()])

        return train_handle, test_handle

    def _sample_observations(self, keys: typing.Iterable[int]) -> np.ndarray:
        if len(self.all_data.shape)>1:
            # all_data is an np.ndarray already loaded into the memory
            if self.dataset=='mnist':
                obs = self.all_data[keys]
                paddings = ((0,0),(2,2), (2,2), (0,0))
                obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
            elif self.dataset=='shifted_mnist':
                obs = []
                keys = list(keys)
                for key in keys:
                    obs.append(_shift_mnist_np(self.all_data[key]))
                obs = np.stack(obs)
            elif self.dataset=='shifted_3pos_mnist':
                obs = []
                keys = list(keys)
                for key in keys:
                    obs.append(_shift_mnist_3pos_np(self.all_data[key]))
                obs = np.stack(obs)
            elif self.dataset=='rotated_mnist':
                obs = []
                keys = list(keys)
                for key in keys:
                    obs.append(_rotate_mnist_np(self.all_data[key]))
                obs = np.stack(obs)
            else:
                obs = self.all_data[keys]
            return obs
        else:
            # all_data is a 1d array of paths
            obs = []
            keys = list(keys)
            for key in keys:
                img = self._read_image(key)
                obs.append(img)
            return np.stack(obs)

    def _read_image(self, key: int) -> np.ndarray:
        seed = 123
        assert key==int(os.path.split(self.all_data[key])[1][:-4])-1, 'Mismatch between key and img_file_name'
        if self.dataset == 'celebA':
            point = self._read_celeba_image(self.all_data[key])
        else:
            raise Exception('Disc read for {} not implemented yet...'.format(self.dataset))

        return point

    def _read_celeba_image(self, file_path: str) -> np.ndarray:
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(file_path)
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height)/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        im_array = np.array(im).reshape(datashapes['celebA']) / 255.
        im.close()
        return im_array
