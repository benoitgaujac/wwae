from math import pi
import tensorflow.compat.v1 as tf
import numpy as np
import math as m

from utils import get_batch_size
from ops._ops import logsumexp
# from sw import sw
from kymatio.scattering2d.filter_bank import filter_bank
from wemd import wemd
from wgan import wgan, wgan_v2

import pdb
import typing

######### latent losses #########
def kl_penalty(pz_mean: tf.Tensor, pz_sigma: tf.Tensor, encoded_mean: tf.Tensor, encoded_sigma: tf.Tensor) -> tf.Tensor:
    """
    Compute KL divergence between prior and variational distribution
    """
    kl = encoded_sigma / pz_sigma \
        + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
        + tf.log(pz_sigma) - tf.log(encoded_sigma)
    kl = 0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)

def cross_entropy_loss(opts: dict, inputs: tf.Tensor, dec_mean: tf.Tensor, dec_Sigma: tf.Tensor) -> tf.Tensor:
    if opts['decoder']=='gaussian':
        cross_entropy = tf.log(2*pi) + tf.log(dec_Sigma) + tf.square(inputs-dec_mean) / dec_Sigma
        cross_entropy = -0.5 * tf.reduce_sum(cross_entropy,axis=-1)
    else:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=dec_mean)
        cross_entropy = tf.reduce_sum(cross_entropy,axis=-1)
    return cross_entropy

def mmd_penalty(opts: dict, sample_qz: tf.Tensor, sample_pz: tf.Tensor) -> tf.Tensor:
    """
    Comput MMD latent penalty
    """
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2, tf.int32)
    distances_pz = square_dist(sample_pz, sample_pz)
    distances_qz = square_dist(sample_qz, sample_qz)
    distances = square_dist(sample_qz, sample_pz)

    if opts['mmd_kernel'] == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        # sigma2_k = opts['latent_space_dim'] * sigma2_p
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif opts['mmd_kernel'] == 'IMQ':
        Cbase = 2 * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    elif opts['mmd_kernel'] == 'RQ':
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            res1 = (1. + distances_qz / scale / 2.) ** (-scale)
            res1 += (1. + distances_pz / scale / 2.) ** (-scale)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = (1. + distances / scale / 2.) ** (-scale)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2

    return stat

def square_dist(sample_x: tf.Tensor, sample_y: tf.Tensor) -> tf.Tensor:
    """
    Wrapper 2 to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


######### rec losses #########
def wae_ground_cost(opts: dict,  x1: tf.Tensor, x2: tf.Tensor, is_training=False, reuse=False) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the WAE's ground cost
    x1: image data             [batch,h,w,c]
    x2: image reconstruction   [batch,h,w,c]
    return batch reconstruction cost [batch,]
    """
    critic_reg = None
    intensities_reg = 0.
    # Compute chosen cost
    if opts['cost'] == 'l2':
        cost = l2_cost(x1, x2)
    elif opts['cost'] == 'l2sq':
        cost = l2sq_cost(x1, x2)
    elif opts['cost'] == 'l2sq_norm':
        cost = l2sq_norm_cost(x1, x2)
    elif opts['cost'] == 'l1':
        cost = l1_cost(x1, x2)
    # elif opts['cost'] == 'sw':
    #     cost = sw(opts,  x1, x2, reuse=reuse)
    elif opts['cost'] == 'wemd':
        h,w,c = x1.get_shape().as_list()[1:]
        J = int(np.log2(h))
        cost, intensities_reg = wemd(opts, x1, x2)
    elif opts['cost'] == 'wgan':
        cost, intensities_reg, critic_reg = wgan(opts, x1, x2, is_training=is_training, reuse=reuse)
        critic_reg = tf.reduce_mean(critic_reg)
    elif opts['cost'] == 'wgan_v2':
        cost, intensities_reg, critic_reg = wgan_v2(opts, x1, x2, is_training=is_training, reuse=reuse)
        critic_reg = tf.reduce_mean(critic_reg)
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    return cost, intensities_reg, critic_reg

def l2_cost(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    # c(x,y) = ||x - y||_2
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=[-3,-2,-1])
    cost = tf.sqrt(1e-10 + cost)
    # return tf.reduce_mean(cost)
    return cost

def l2sq_cost(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    # c(x,y) = sum_i(||x - y||_2^2[:,i])
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=[-3,-2,-1])
    # return tf.reduce_mean(cost)
    return cost

def l2sq_norm_cost(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    # c(x,y) = mean_i(||x - y||_2^2[:,i])
    cost = tf.reduce_mean(tf.square(x1 - x2), axis=[-3,-2,-1])
    # return tf.reduce_mean(cost)
    return cost

def l1_cost(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    # c(x,y) = ||x - y||_1
    cost = tf.reduce_sum(tf.abs(x1 - x2), axis=[-3,-2,-1])
    # return tf.reduce_mean(cost)
    return cost
