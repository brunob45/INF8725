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
    # create the difference of gaussian for a full octave
    for i in range(1, len(imgs)):
        octave.append(normalizeDoG(imgs[i], imgs[i-1]))

    return (octave, imgs)

def differenceDeGaussiennes(image_initiale, s, nb_octave):
    diffs = []
    imgs = []

    img = image_initiale
    # get the full DoG pyramid
    for _ in range(0,nb_octave):
        (octave, i) = get_dog_octave(img, s)
        diffs.append(octave)
        imgs.append(i)
        # downscale the image
        img = resize(i[s])

    return (diffs, imgs)

def normalizeDoG(img,cpy):
    # normalize the value of the dog to find extrema (otherwise, the minimums and maximums are all in the scales above or below)
    dog = img-cpy
    return (dog-np.min(dog))/(np.max(dog)-np.min(dog))

if __name__ == '__main__':
    img = openImage('droite.jpg')

    scale = 3
    octave = 2

    (results, _) = differenceDeGaussiennes(img, scale, octave)

    plt.plot([octave,scale])

    for j in range(0,octave):
        for i in range(0,scale):
            img = results[j][i]
            plt.subplot(octave, scale, 1 + j*scale +i)
            plt.title(1.6*2**(i/scale+j))
            show(img)

    plt.show()
