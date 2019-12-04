#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def applyGaussian(img, scale, sigma=1.0):
    return gaussian_filter(img, sigma=sigma * 2**(1/scale))

def get_octave(img, scale, sigma=1.0):
    octave = [img]
    for s in range(1, scale+3):
        octave.append(applyGaussian(octave[-1], scale, sigma))
    return octave

if __name__ == '__main__':
    img = openImage('Lenna.jpg')
    scale = 4
    octave = 3
    sigma = 1.0

    plt.plot([octave,scale])

    for j in range(0, octave):
        o = get_octave(img, scale, sigma)

        for i in range(scale):
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(sigma * 2**(i/scale+j))
            show(o[i])
        img = resize(o[scale])

    plt.show()