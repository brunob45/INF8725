#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def applyGaussian(img, i, s):
    return gaussian_filter(img, sigma=2**(i/s))


if __name__ == '__main__':
    img = openImage('droite.jpg')
    scale = 3
    octave = 2

    plt.plot([octave,scale])

    for j in range(0,octave):
        tmp = img
        for i in range(0,scale):
            tmp = applyGaussian(tmp, i, scale)
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(2**(i/scale+j))
            show(tmp)
        img = resize(img)

    plt.show()