#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

<<<<<<< HEAD
def applyGaussian(img, s):
    sigma = 1.6
    g = sigma*(2**(1/(s-1)))
    return gaussian_filter(img, g)
    # return gaussian_filter(img, sigma=0.7950)
=======
>>>>>>> 14ee4bcc31aa708d8cacbd506de45cb91587e775

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
            plt.title((j+1)*2**(i/scale))
            show(tmp)
        img = resize(img)

    plt.show()