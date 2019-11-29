#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def applyGaussian(img, s):
    return gaussian_filter(img, sigma=2**(1/(s-1)))
    # return gaussian_filter(img, sigma=0.7950)

    filtre = np.array([[1/16, 1/8, 1/16],
                       [1/8,  1/4, 1/8 ],
                       [1/16, 1/8, 1/16]])
    return signal.convolve2d(img, filtre, mode='same')


if __name__ == '__main__':
    img = openImage('Lenna.jpg')
    scale = 6
    octave = 4

    plt.plot([octave,scale])

    for j in range(0,octave):
        plt.subplot(octave, scale, 1 + j*scale)
        show(img)

        for i in range(1,scale):
            img = applyGaussian(img, scale)
            plt.subplot(octave, scale, 1 + j*scale +i)
            show(img)

    plt.show()