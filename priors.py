import sys
import time
import os
from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import pdb
import typing


def init_prior(opts: dict) -> typing.Tuple[np.ndarray, np.ndarray]:
    Sigma = opts['pz_sigma']**2*np.ones(opts['zdim'], dtype='float32')
    if opts['prior']=='gaussian':
        means = np.zeros(opts['zdim'], dtype='float32')
    elif opts['prior']=='gmm':
        means = gmm(opts['pz_nmix'], opts['pz_sigma'])
    else:
        raise ValueError('Unknown {} prior' % opts['prior'])

    return means, Sigma

def gmm(nmix, sigma: np.ndarray) -> np.ndarray:
    """
    Initialize the means of the GMM on the diagonal

    return nmix 2d points on the diagonal separated by 4*sigma
    """
    stop = sqrt(2) * sigma * (nmix + 1)
    xx = np.linspace(-stop, stop, nmix+2, dtype='float32')
    means = np.stack([xx,xx], axis=-1)

    return means[1:-1]
