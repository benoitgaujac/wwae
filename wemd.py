import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sio
from kymatio.scattering2d.filter_bank import filter_bank

import typing


'''
def wemd(x1,x2):

    h, w, c = x1.get_shape().as_list()[1:]

    mass1 = tf.reduce_sum(x1, axis=[1,2]) #[b,c]
    xs1 = x1 / tf.reshape(mass1, [-1,1,1,c])
    mass2 = tf.reduce_sum(x2, axis=[1,2]) #[b,c]
    xs2 = x2 / tf.reshape(mass2, [-1,1,1,c])


    d = xs1 - xs2
    d = tf.cast(d, tf.complex64)
    d = tf.transpose(d, [0,3,1,2])  # (b,c,h,w)

    matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N64_J6_L8.mat')
    J = 6
    L = 16
    fftphi = matfilters['filt_fftphi'].astype(np.complex_)
    fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)  # (J,L,h,w)

    fftd = tf.signal.fft2d(d)
    fftd = tf.reshape(fftd, [-1,c,1,1,h,w])

    fftpsi = tf.cast(tf.reshape(fftpsi, [1,1,J,L,h,w]), tf.complex64)

    hatprod = fftpsi*fftd

    prod = tf.signal.ifft2d(hatprod)

    mod = tf.math.abs(prod)  # (b,c,J,L,h,w)
    mod_mean = tf.reduce_mean(mod, axis=[-5,-3,-2,-1])

    coef = [2**(j/2) for j in [4,3,2,1,0]]
    coef = tf.reshape(coef, [1,-1])
    coef = tf.cast(coef, tf.float32)

    wemd = tf.reduce_mean(1*mod_mean, axis=-1)

    diff_m = (1. - mass2/mass1)**2
    diff_m = tf.math.reduce_mean(diff_m, axis=-1)

#    print(diff_m.shape, wemd.shape)

    return wemd #+ 1e-12*diff_m
'''

def wemd(opts: dict, x1: tf.Tensor, x2: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:

    h, w, c = x1.get_shape().as_list()[1:]
    J = int(np.log2(h))  #number of scales
    L = opts['orientation_num'] #number of orientations
    # Normalising inputs
    mass1 = tf.reduce_sum(x1, axis=[1,2], keepdims=True) #[b,1,1,c]
    xs1 = x1 / mass1
    mass2 = tf.reduce_sum(x2, axis=[1,2], keepdims=True) #[b,1,1,c]
    xs2 = x2 / mass2
    # Difference probability
    d = xs1 - xs2
    d = tf.cast(d, tf.complex64)
    d = tf.transpose(d,[0,3,1,2])  # (b,c,h,w)
    # Fourrier transform
    fftd = tf.signal.fft2d(d)
    # Get waves filters
    dictt = filter_bank(h, w, J, L)['psi']
    waves = np.zeros([J, L, h, w])
    for j in range(J):
        for theta in range(L):
            waves[j,theta,:,:] = dictt[L*j+theta][0]
    waves = tf.cast(waves, tf.complex64)
    # Conv in Fourrier domain
    hatprod = tf.reshape(waves, [1,1,J,L,h,w])*tf.reshape(fftd, [-1,c,1,1,h,w]) #(b,c,J,L,h,w)
    prod = tf.signal.ifft2d(hatprod) #(b,c,J,L,h,w)
    # Module of wave coeffs
    mod = tf.math.abs(prod) #(b,c,J,L,h,w)
    mod = tf.reduce_sum(mod, axis=[-3,-2,-1]) #(b,c,J)
    # coef for putting different weights to different scales as in the paper
    # coef = tf.convert_to_tensor([2**(j/2) for j in range(J-1,-1,-1)],dtype=tf.float32)
#    coef = tf.convert_to_tensor([2**(j) for j in range(J-1,-1,-1)],dtype=tf.float32)
#    coef = tf.reshape(coef,[1,1,J])
    # wemd
    wemd = tf.reduce_sum(mod, axis=-1) #(b,c)
    # intensities reg
    diff_m = (1. - tf.reshape(mass2/mass1, [-1,c]))**2 #(b,c)

    return tf.reduce_mean(wemd, axis=-1), tf.reduce_mean(diff_m, axis=-1)
