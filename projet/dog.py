#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gaussian import applyGaussian

def DoG(img, s, nb_octave):
    diffs = []
    for octave in range(0,nb_octave):
        previous = img
        for scale in range(0,s):
            current = applyGaussian(img, scale, s)
            diffs.append(previous - current)   #difference of Gaussian
            previous = current
        img = resize(img)
    return diffs


if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    scale = 3
    octave = 2

    results = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for j in range(0,octave):
        for i in range(0,scale):
            img = results[i + j * scale]
            plt.subplot(octave, scale, 1 + j*scale +i)
            show(img)

    plt.show()
