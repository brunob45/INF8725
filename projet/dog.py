#!/usr/bin/env python3

from imgproc import openImage, show, resize

from scipy import signal
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def applyGaussian(img):
    # return gaussian_filter(img, sigma=0.7950)

    filtre = np.array([[1/16, 1/8, 1/16],
                       [1/8,  1/4, 1/8 ],
                       [1/16, 1/8, 1/16]])
    return signal.convolve2d(img, filtre, mode='same')


def DoG(img, s, nb_octave):
    results = []
    imgcpy = img.copy()
    for octave in range(nb_octave):
        for scale in range(s):
            imgcpy = applyGaussian(img)
            results.append(img - imgcpy)   #original line to get difference of Gaussian
            #results.append(imgcpy)              #test line to get filtered images
            img = imgcpy
        img = resize(img)
    return results


if __name__ == '__main__':
    #img = openImage('droite.jpg')
    img = openImage('Lenna.jpg')
    
    img = img / np.max(img)

    #results = DoG(img, 3, 2)
    results = DoG(img, 6, 1)
    plt.plot([2,3])
    plt.subplot(231)
    show(results[0])
    plt.subplot(232)
    show(results[1])
    plt.subplot(233)
    show(results[2])
    plt.subplot(234)
    show(results[3])
    plt.subplot(235)
    show(results[4])
    #plt.subplot(236)
    #show(results[5])
    plt.show()

    #plt.plot([1,3])
    #plt.subplot(131)
    #show(results[0])
    #plt.subplot(132)
    #show(results[1])
    #plt.subplot(133)
    #show(results[2])
    #plt.show()