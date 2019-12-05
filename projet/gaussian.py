#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def applys0(img, s0):
    # apply the first gaussian filter
    return gaussian_filter(img, sigma=s0)

def applyGaussian(img, scale):
    # apply following gaussian filter
    return gaussian_filter(img, sigma=2**(1/scale))

def get_octave(img, scale):
    # apply the gaussian filter for a full octave
    octave = [img]
    for s in range(1, scale+3):
        octave.append(applyGaussian(octave[-1], scale))
    return octave

if __name__ == '__main__':
    img = openImage('droite.jpg')

    sigma = 1.6
    img = applys0(img, sigma)

    scale = 3
    octave = 2

    plt.plot([octave,scale])

    for j in range(0, octave):
        o = get_octave(img, scale)

        for i in range(scale):
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(sigma * 2**(i/scale+j))
            show(o[i])
        img = resize(o[scale])

    plt.show()