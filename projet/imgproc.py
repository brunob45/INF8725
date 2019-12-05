#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# This file contain utilities to open images, show them with imshow and resize an array

def openImage(filename):
    img = Image.open(filename).convert('L')
    return img / np.max(img)

def show(img):
    if type(img) is Image.Image:
        img = np.array(img)
    plt.imshow(img, cmap='gray')

def resize(myImg):
    if type(myImg) is np.ndarray:
        myImg = Image.fromarray(myImg)
    return np.array(myImg.resize((int(myImg.width/2), int(myImg.height/2)), Image.ANTIALIAS))

if __name__ == '__main__':
    img = openImage('Lenna.jpg')
    
    plt.plot([1,3])

    plt.subplot(1, 3, 1)
    show(img)

    plt.subplot(1, 3, 2)
    img = resize(img)
    show(img)

    plt.subplot(1, 3, 3)
    img = resize(img)
    show(img.rotate(10))

    plt.show()