#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gaussian import applyGaussian

def DoG(img, s, nb_octave):
    results = []
    imgs = []
    imgcpy = img.copy()
    for octave in range(0,nb_octave):
        for scale in range(0,s):
            imgs.append(img)
            imgcpy = applyGaussian(img, s)
            dog = normalizeDoG(img,imgcpy)
            results.append(dog)   #original line to get difference of Gaussian
            #results.append(imgcpy)              #test line to get filtered images
            img = imgcpy
        img = resize(img)
    return results, imgs

def normalizeDoG(img,cpy):
    dog = img-cpy
    return (dog-np.min(dog))/(np.max(dog)-np.min(dog))

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    scale = 3
    octave = 2

    results,imgs = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for j in range(0,octave):
        for i in range(0,scale):
            img = results[i + j * scale]
            plt.subplot(octave, scale, 1 + j*scale +i)
            show(img)

    plt.show()
