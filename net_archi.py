import numpy as np
import tensorflow.compat.v1 as tf

from ops.linear import Linear
from ops.batchnorm import Batchnorm_layers
from ops.conv2d import Conv2d
from ops.deconv2d import Deconv2D
from ops.resnet import ResidualBlock, OptimizedResBlockEnc1
import ops._ops


#################################### Encoder/Decoder ####################################

######### mlp #########
def mlp_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    layer_x = input
    # hidden 0
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid2/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def mlp_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    layer_x = input
    # hidden 0
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid2/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(
            opts, layer_x, 'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), init=opts['mlp_init'], scope='hid_final')

    return outputs

######### mnist/svhn #########
def mnist_conv_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def mnist_conv_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse: bool, is_training: bool) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                8*8*128, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 128])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                32]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

######### 64x64 transformed mnist #########
def mnist_64_conv_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def mnist_64_conv_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse: bool, is_training: bool) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                16*16*128, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 16, 16, 128])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                32]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

######### cifar10 #########
def cifar10_conv_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def cifar10_conv_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse: bool, is_training: bool) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    w,h = output_dim[0], output_dim[1]
    layer_x = input
    # Linear layers
    _out_shape = [int(w/2**3), int(h/2**3), 512]
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                256]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                128]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid3/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

def cifar10_resnet_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:

    layer_x = input
    # optimized resblock
    layer_x = OptimizedResBlockEnc1(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'],
                                filter_size=3, scope='optrb',
                                init=opts['conv_init'])
    # resblock 1
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb1', init=opts['conv_init'],
                                resample='down', is_training=is_training,
                                reuse=reuse)
    # resblock 2
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb2', init=opts['conv_init'],
                                resample=None, is_training=is_training,
                                reuse=reuse)
    # resblock 3
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb3', init=opts['conv_init'],
                                resample=None, is_training=is_training,
                                reuse=reuse)
    # final layer
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # layer_x = tf.reduce_mean(layer_x, axis=[1,2])
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def cifar10_resnet_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:

    w,h = output_dim[0], output_dim[1]
    layer_x = input

    # Linear layer
    _out_shape = [int(w/2**3), int(h/2**3), opts['zdim']]
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
    # resblock 1
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb1', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # resblock 2
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb2', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # resblock 3
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb3', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # final layers
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid_final/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    output = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=output_dim[-1], filter_size=3,
                                stride=1, scope='hid_final/conv',
                                init=opts['conv_init'])

    return output

######### celebA #########
def celebA_conv_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def celebA_conv_decoder(opts: dict, input: tf.Tensor, output_dim: int, reuse: bool, is_training: bool) -> tf.Tensor:
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    w,h = output_dim[0], output_dim[1]
    layer_x = input
    # Linear layers
    _out_shape = [int(w/2**4), int(h/2**4), 512]
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                256]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                128]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid3/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 4
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                32]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid4/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid4/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

def celebA_resnet_encoder(opts: dict, input: tf.Tensor, output_dim: int, reuse=False, is_training=False) -> tf.Tensor:

    layer_x = input
    # first conv layer
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                stride=1, scope='hid0/conv',
                                init=opts['conv_init'])
    # resblock 1
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=2*opts['zdim'], filter_size=3,
                                scope='rb1', init=opts['conv_init'],
                                resample='down', is_training=is_training,
                                reuse=reuse)
    # resblock 2
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=4*opts['zdim'], filter_size=3,
                                scope='rb2', init=opts['conv_init'],
                                resample='down', is_training=is_training,
                                reuse=reuse)
    # resblock 3
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=8*opts['zdim'], filter_size=3,
                                scope='rb3', init=opts['conv_init'],
                                resample='down', is_training=is_training,
                                reuse=reuse)
    # resblock 4
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=8*opts['zdim'], filter_size=3,
                                scope='rb4', init=opts['conv_init'],
                                resample='down', is_training=is_training,
                                reuse=reuse)
    # final layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def celebA_resnet_decoder(opts: dict, input: tf.Tensor, output_dim, reuse=False, is_training=False) -> tf.Tensor:

    w,h = output_dim[0], output_dim[1]
    layer_x = input

    # Linear layer
    _out_shape = [int(w/2**4), int(h/2**4), 8*opts['zdim']]
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
    # resblock 1
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=8*opts['zdim'], filter_size=3,
                                scope='rb1', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # resblock 2
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=4*opts['zdim'], filter_size=3,
                                scope='rb2', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # resblock 3
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=2*opts['zdim'], filter_size=3,
                                scope='rb3', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # resblock 4
    layer_x = ResidualBlock(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=opts['zdim'], filter_size=3,
                                scope='rb4', init=opts['conv_init'],
                                resample='up', is_training=is_training,
                                reuse=reuse)
    # final layers
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid_final/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    output = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=output_dim[-1], filter_size=3,
                                stride=1, scope='hid_final/conv',
                                init=opts['conv_init'])

    return output


net_archi = {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
            'conv': {'mnist':{'encoder': mnist_conv_encoder, 'decoder': mnist_conv_decoder},
                    'transformed_mnist':{'encoder': mnist_64_conv_encoder, 'decoder': mnist_64_conv_decoder},
                    'svhn':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
                    'cifar10':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
                    'celebA':{'encoder': celebA_conv_encoder, 'decoder': celebA_conv_decoder}},
            'resnet': {'celebA':{'encoder': celebA_resnet_encoder, 'decoder': celebA_resnet_decoder},
                    'cifar10':{'encoder': cifar10_resnet_encoder, 'decoder': cifar10_resnet_decoder}}
            }


#################################### Critic ####################################

######### mlp #########
def mlp_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = tf.keras.layers.Flatten(inputs)
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid0/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid1/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid2/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 3
        layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid3/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    np.prod(in_shape),
                                    init=opts['mlp_init'],
                                    scope='hid_final')

    return outputs

######### conv #########
def singleconv_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # conv
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v2_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=3,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v3_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v4_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=64, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=256, filter_size=4,
                                    stride=1, scope='hid2/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 3
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=512, filter_size=4,
                                    stride=1, scope='hid3/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v5_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=256, filter_size=3,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=512, filter_size=3,
                                    stride=1, scope='hid2/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

######### convdeconv #########
def convdeconv_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=2, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,in_shape[0],in_shape[1],128],
                                    filter_size=4,
                                    stride=2, scope='hid1/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+in_shape, filter_size=1,
                                    stride=1, scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def convdeconv_v2_critic(opts, inputs, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=64, filter_size=4,
                                    stride=2, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,in_shape[0],in_shape[1],32],
                                    filter_size=4,
                                    stride=2, scope='hid1/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+in_shape, filter_size=1,
                                    stride=1, scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def convdeconv_v3_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=32, filter_size=3,
                                    stride=2, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,in_shape[0],in_shape[1],64],
                                    filter_size=3,
                                    stride=2, scope='hid1/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+in_shape, filter_size=1,
                                    stride=1, scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def resnet_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # # skip connection
        # layer_x += inputs
        # layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs + inputs

def resnet_v2_critic(opts: dict, inputs: tf.Tensor, scope=None, is_training=False, reuse=False) -> tf.Tensor:
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        # skip connection
        shortcut = Conv2d(opts, inputs, inputs.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='skip/conv',
                                    init=opts['conv_init'])
        layer_x += shortcut
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs + inputs


critic_archi = {'mlp': mlp_critic,
            'singleconv': singleconv_critic,
            'conv': conv_critic,
            'conv_v2': conv_v2_critic,
            'conv_v3': conv_v3_critic,
            'conv_v4': conv_v4_critic,
            'conv_v5': conv_v5_critic,
            'convdeconv': convdeconv_critic,
            'convdeconv_v2': convdeconv_v2_critic,
            'convdeconv_v3': convdeconv_v3_critic,
            'resnet': resnet_critic,
            'resnet_v2': resnet_v2_critic
            }
