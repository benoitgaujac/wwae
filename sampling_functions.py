import sys
import time
import os
from math import sqrt, cos, sin, pi, ceil
import numpy as np
import tensorflow.compat.v1 as tf
from scipy import ndimage

import pdb
import typing

def sample_pz(opts: dict, means: tf.Tensor, Sigma: tf.Tensor, batch_size=100) -> tf.Tensor:
    if opts['prior']=='gaussian' or opts['prior']=='implicit':
        noise = sample_gaussian(means, Sigma, batch_size, 'numpy')
    elif opts['prior']=='gmm':
        noise = sample_gmm(opts['pz_nmix'], means, Sigma, batch_size)
    elif opts['prior']=='dirichlet':
        noise = sample_dirichlet(means, batch_size)
    else:
        assert False, 'Unknown prior %s' % opts['prior']
    return noise

def sample_gaussian(means: tf.Tensor, Sigma: tf.Tensor, batch_size=100, type='numpy') -> tf.Tensor:
    """
    Sample noise from gaussian distribution with parameters
    means and covs
    """
    if type =='tensorflow':
        shape = tf.shape(means)
        eps = tf.random_normal(shape, dtype=tf.float32)
        noise = means + tf.multiply(eps, tf.sqrt(1e-10+Sigma))
    elif type =='numpy':
        if len(np.shape(means))<2:
            shape = (batch_size,)+np.shape(means)
        else:
            shape = np.shape(means)
        eps = np.random.normal(0., 1., shape).astype(np.float32)
        noise = means + np.multiply(eps, np.sqrt(1e-10+Sigma))
    return noise

def sample_gmm(nmix: int, means: tf.Tensor, Sigma: tf.Tensor, batch_size=100) -> tf.Tensor:
    """
    Sample prior noise according to sampling_mode
    """

    noises = sample_all_gmm(means, Sigma, batch_size)
    mixtures_id = np.random.randint(nmix, size=batch_size)
    samples = noises[np.arange(batch_size), mixtures_id]

    return samples

def sample_all_gmm(means: tf.Tensor, Sigma: tf.Tensor, batch_size=100) -> tf.Tensor:
    """
    Sample for each component of the gmm

    means: [K,zdim]
    Sigma: [zdim]
    """

    shape = means.shape
    eps = np.random.normal(0.,1.,(batch_size,)+shape).astype(np.float32)
    noise = means + np.multiply(eps,np.sqrt(1e-10+Sigma))

    return noise

def sample_dirichlet(alpha: tf.Tensor, batch_size=100) -> tf.Tensor:
    """
    Sample noise from dirichlet distribution with parameters
    alpha
    """
    return np.random.dirichlet(alpha, batch_size)

def sample_unif(shape: typing.Iterable[int], minval=0, maxval=None, dtype=tf.float32) -> tf.Tensor:
    """
    Sample noise from Unif[minval,maxval]
    """
    return tf.random.uniform(shape, minval, maxval, dtype)

def sample_bernoulli(params: object) -> tf.Tensor:
    """
    Sample noise from Bernoulli distribution with mean parameters
    params
    """
    assert False, 'tfp not available on cluster gpu yet'
    """
    shape = tf.shape(params)
    bernoulli_dist = tfp.distributions.Bernoulli(logits=params, dtype=tf.float32)
    return bernoulli_dist.sample()
    """

def traversals(anchors: np.ndarray, nsteps: int, std=1.0) -> np.ndarray:
    """
    Genereate linear grid space
        - anchors[nanchors,zdim]: encoding
        - nsteps:  Num of steps in the interpolation
    Return:
    linespce[nanchors,nsteps,zdim]: list of linear interpolations for each latent dim
    """
    nanchors = np.shape(anchors)[0]
    linespce = []
    std_range = std
    start = anchors-std_range
    stop = anchors+std_range
    for n in range(nanchors):
        int_m = np.linspace(start[n],anchors[n],int(nsteps/2.),endpoint=False)
        if nsteps%2==0:
            int_p = np.linspace(anchors[n],stop[n],int(nsteps/2.),endpoint=True)
        else:
            int_p = np.linspace(anchors[n],stop[n],int(nsteps/2.)+1,endpoint=True)
        linespce.append(np.concatenate((int_m,int_p)))
    linespce = np.stack(linespce)

    return linespce

def interpolations(anchors: np.ndarray, nsteps: int, std=1.0) -> np.ndarray:
    """
    Genereate linear grid space
        - anchors[nanchors,2,zdim]: encoding
        - nsteps:  Num of steps in the interpolation
    Return:
    linespce[nanchors,nsteps,zdim]: list of linear interpolations for each latent dim
    """
    nanchors = np.shape(anchors)[0]
    linespce = []
    start = anchors[:,0]
    stop = anchors[:,1]
    for n in range(nanchors):
        inter = np.linspace(start[n],stop[n], nsteps, endpoint=True)
        linespce.append(inter)
    linespce = np.stack(linespce)

    return linespce

def grid(nsteps: int, zdim: int) -> np.ndarray:
    """
    Generate a 2D grid of nsteps x nsteps in [-2,2]**2
    return grid: [nsteps,nsteps,zdim]
    """
    assert zdim==2, "latent dimension must be equal to 2"
    linespace = np.linspace(-2., 2., nsteps)
    xv, yv = np.meshgrid(linespace,linespace)
    grid = np.stack((xv,yv),axis=-1)

    return grid

def shift(opts, inputs: np.ndarray, shift_dir: np.ndarray, shift: float) -> np.ndarray:
    ninputs = inputs.shape[0]
    in_shape = np.array(inputs.shape[1:-1])
    # padded = np.pad(inputs, ((0,0),(shift,shift),(shift,shift),(0,0)), mode='edge')
    padded = np.pad(inputs, ((0,0),(shift,shift),(shift,shift),(0,0)), mode='linear_ramp',end_values=0)
    # padded = np.pad(inputs, ((0,0),(shift,shift),(shift,shift),(0,0)), mode='mean')
    start = shift_dir*shift + shift
    end = in_shape.reshape(-1,2) + shift_dir*shift + shift
    # idx_w = np.linspace(start[:,0],end[:,0], in_shape[0],dtype=np.int32).transpose()
    # idx_h = np.linspace(start[:,1],end[:,1], in_shape[1],dtype=np.int32).transpose()
    shifted = []
    for n in range(ninputs):
        shifted.append(padded[n,start[n,0]:end[n,0],start[n,1]:end[n,1]])
        print(shifted[-1].shape)
    return np.stack(shifted,axis=0)

def rotate(opts, batch: np.ndarray, rot_dir: np.ndarray, nangle: int, base_angle: float) -> np.ndarray:
    angle = rot_dir * base_angle * nangle
    batch_size = batch.shape[0]
    rotated = []
    for n in range(batch_size):
        rotated.append(ndimage.rotate(batch[n], angle[n], reshape=False))

    return np.stack(rotated,axis=0)
