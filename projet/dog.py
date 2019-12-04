#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gaussian import applys0, get_octave

def get_dog_octave(img, scale):
    octave = []
    imgs = get_octave(img, scale)

    for i in range(1, len(imgs)):
        octave.append(normalizeDoG(imgs[i], imgs[i-1]))
        # octave.append(imgs[i] - imgs[i-1])   #difference of Gaussian

    return (octave, imgs)

def DoG(img, scale, nb_octave, sigma=1.6):
    diffs = []
    imgs = []

    img = applys0(img, sigma)

    for _ in range(0,nb_octave):
        (octave, i) = get_dog_octave(img, scale)
        diffs.append(octave)
        imgs.append(i)

        img = resize(i[scale])

    return (diffs, imgs)

def normalizeDoG(img,cpy):
    dog = img-cpy
    return (dog-np.min(dog))/(np.max(dog)-np.min(dog))

if __name__ == '__main__':
    img = openImage('droite.jpg')

    scale = 3
    octave = 2

    (results, _) = DoG(img, scale, octave, 1.6)

    plt.plot([octave,scale])

    for j in range(0,octave):
        for i in range(0,scale):
            img = results[j][i]
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(1.6*2**(i/scale+j))
            show(img)

    plt.show()
