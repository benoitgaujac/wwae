import numpy as np
import tensorflow.compat.v1 as tf

from math import pi

from datahandler import datashapes
from sampling_functions import sample_gaussian
from net_archi import net_archi , critic_archi

import logging
import pdb
import typing

def encoder(opts: typing.Dict[str, typing.Any], input: tf.Tensor, output_dim: int, scope=None,
            reuse=False, is_training=False) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.variable_scope(scope, reuse=reuse):
        if opts['net_archi'] == 'mlp':
            encoder = net_archi['mlp']['encoder']
        elif opts['net_archi'] == 'conv' or opts['net_archi'] == 'resnet':
            encoder = net_archi[opts['net_archi']][opts['dataset']]['encoder']
        else:
            raise ValueError('Unknown {} net. archi. for {} dataset'.format(opts['net_archi'],opts['dataset']))
        outputs = encoder(opts, input, output_dim,
                                    reuse,
                                    is_training)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    mean_shape = mean.get_shape().as_list()[1:]
    mean = tf.reshape(mean, [-1,np.prod(mean_shape)])
    Sigma_shape = Sigma.get_shape().as_list()[1:]
    Sigma = tf.reshape(Sigma, [-1,np.prod(Sigma_shape)])

    if opts['encoder'] == 'det':
        z = mean
    elif opts['encoder'] == 'gauss':
        z = sample_gaussian(mean, Sigma, type='tensorflow')
    else:
        assert False, 'Unknown encoder %s' % opts['encoder']

    return z, mean, Sigma


def decoder(opts: typing.Dict[str, typing.Any], input: tf.Tensor, output_dim: tf.Tensor, scope=None,
            reuse=False, is_training=False) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.variable_scope(scope, reuse=reuse):
        if opts['net_archi'] == 'mlp':
            decoder = net_archi['mlp']['decoder']
        elif opts['net_archi'] == 'conv' or opts['net_archi'] == 'resnet':
            decoder = net_archi[opts['net_archi']][opts['dataset']]['decoder']
        else:
            raise ValueError('Unknown {} net. archi. for {} dataset'.format(opts['net_archi'],opts['dataset']))
        outputs = decoder(opts, input, output_dim,
                                    reuse,
                                    is_training)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    mean_shape = mean.get_shape().as_list()[1:]
    mean = tf.reshape(mean, [-1,np.prod(mean_shape)])
    Sigma_shape = Sigma.get_shape().as_list()[1:]
    Sigma = tf.reshape(Sigma, [-1,np.prod(Sigma_shape)])

    # sampling from gaussian if needed
    if opts['decoder'] == 'gaussian':
        x = sample_gaussian(mean, Sigma, type='tensorflow')
    else:
        x = mean
    # normalise to [0,1] or [-1,1]
    if opts['input_normalize_sym']:
        x = tf.nn.tanh(x)
    else:
        x = tf.nn.sigmoid(x)

    x = tf.reshape(x, [-1] + datashapes[opts['dataset']])

    return x, mean, Sigma


def critic(opts: typing.Dict[str, typing.Any], inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    """
    Critic network of the w1
    inputs: [batch,w,h,c]
    outputs: [batch,w,h,c]
    """
    in_shape = inputs.get_shape().as_list()[1:]
    if opts['wgan_critic_archi']=='coef':
        layer_x = inputs
        with tf.variable_scope(scope, reuse=reuse):
                coef = tf.get_variable("W", in_shape, tf.float32,
                        tf.random_normal_initializer(stddev=opts['init_std']))
                # element-wise multi
                outputs = layer_x*coef
    else:
        critic = critic_archi[opts['wgan_critic_archi']]
        outputs = critic(opts, inputs, scope, is_training, reuse)

    return tf.reshape(outputs, [-1,]+in_shape)
