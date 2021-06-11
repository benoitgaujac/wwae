import numpy as np
import tensorflow as tf

import functools

import pdb

from ops.linear import Linear
from ops.batchnorm import Batchnorm_layers
from ops.conv2d import Conv2d
import ops._ops


def ConvMeanPool(opts: dict, input: tf.Tensor, input_dim: int, output_dim: int, filter_size: int, scope=None, init='he') -> tf.Tensor:
    output = Conv2d(opts, input, input_dim, output_dim, filter_size, scope=scope, init=init)
    output = tf.nn.avg_pool2d(output, 2, 2, 'VALID')
    return output

def MeanPoolConv(opts: dict, input: tf.Tensor, input_dim: int, output_dim: int, filter_size: int, scope=None, init='he') -> tf.Tensor:
    output = input
    output = tf.nn.avg_pool2d(output, 2, 2, 'VALID')
    output = Conv2d(opts, output, input_dim, output_dim, filter_size, scope=scope, init=init)
    return output

def UpsampleConv(opts: dict, input: tf.Tensor, input_dim: int, output_dim: int, filter_size: int, scope=None, init='he') -> tf.Tensor:
    output = input
    output = tf.keras.layers.UpSampling2D(size=(2,2))(output)
    # output = tf.concat([output, output, output, output], axis=-1) # concat along channel axis
    # output = tf.depth_to_space(output, 2)
    output = Conv2d(opts, output, input_dim, output_dim, filter_size, scope=scope, init=init)
    return output


def ResidualBlock(opts: dict, input: tf.Tensor, input_dim: int, output_dim: int, filter_size: int, scope=None, init='he',
        resample=None, is_training=False, reuse=None) -> tf.Tensor:
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1 = functools.partial(Conv2d, input_dim=input_dim,
                                            output_dim=input_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim,
                                            output_dim=output_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)
        conv_shortcut = MeanPoolConv
    elif resample=='up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim,
                                            output_dim=output_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(Conv2d, input_dim=output_dim,
                                            output_dim=output_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)
    elif resample==None:
        conv_shortcut = Conv2d
        conv_1 = functools.partial(Conv2d, input_dim=input_dim,
                                            output_dim=output_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)
        conv_2 = functools.partial(Conv2d, input_dim=output_dim,
                                            output_dim=output_dim,
                                            filter_size=filter_size,
                                            scope=scope,
                                            init=init)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = input # Identity skip-connection
    else:
        shortcut = conv_shortcut(opts, input=input, input_dim=input_dim, output_dim=output_dim, filter_size=1, scope=scope+'/shortcut',init='normilized_glorot')

    output = input
    output = Batchnorm_layers(opts, output, scope=scope+'/bn0', is_training=is_training, reuse=reuse)
    output = ops._ops.non_linear(output,'relu')
    output = conv_1(opts, input=output, scope=scope+'/conv1', filter_size=filter_size)
    output = Batchnorm_layers(opts, output, scope=scope+'/bn1', is_training=is_training, reuse=reuse)
    output = ops._ops.non_linear(output,'relu')
    output = conv_2(opts, input=output, scope=scope+'/conv2', filter_size=filter_size)

    return shortcut + output

def OptimizedResBlockEnc1(opts: dict, input: tf.Tensor, input_dim: int, output_dim: int, filter_size: int, scope=None, init='he') -> tf.Tensor:
    conv_1  = functools.partial(Conv2d, input_dim=input_dim,
                                        output_dim=output_dim,
                                        filter_size=filter_size,
                                        scope=scope,
                                        init=init)
    conv_2 = functools.partial(ConvMeanPool, input_dim=output_dim,
                                        output_dim=output_dim,
                                        filter_size=filter_size,
                                        scope=scope,
                                        init=init)

    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(opts, input=input, input_dim=input_dim, output_dim=output_dim, filter_size=1, scope=scope+'/shortcut',init=init)

    output = input
    output = conv_1(opts, input=output, scope=scope+'/conv1', filter_size=filter_size)
    output = ops._ops.non_linear(output,'relu')
    output = conv_2(opts, input=output, scope=scope+'/conv2', filter_size=filter_size)
    return shortcut + output
