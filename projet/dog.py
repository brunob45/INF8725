#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gaussian import applyGaussian, get_octave

def get_dog_octave(img, scale, sigma=1.0):
    octave = []
    imgs = get_octave(img, scale, sigma)

    for i in range(1, len(imgs)):
        octave.append(normalizeDoG(imgs[i], imgs[i-1]))
        # octave.append(imgs[i] - imgs[i-1])   #difference of Gaussian

    return (octave, imgs[scale])

def DoG(img, scale, nb_octave):
    diffs = []
    for _ in range(0,nb_octave):
        (octave, img) = get_dog_octave(img, scale)
        diffs.append(octave)
        img = resize(img)
    return diffs

def normalizeDoG(img,cpy):
    dog = img-cpy
    return (dog-np.min(dog))/(np.max(dog)-np.min(dog))

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    scale = 4
    octave = 3

    results = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for j in range(0,octave):
        for i in range(0,scale):
            img = results[j][i]
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(2**(i/scale+j))
            show(img)

    plt.show()
