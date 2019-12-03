#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def openImage(filename):
    img = Image.open(filename).convert('L')
    return img / np.max(img)

def show(img):
    plt.imshow(img, cmap='gray')#, vmin=0, vmax=255)

def resize(myImg):
    if type(myImg) is np.ndarray:
        myImg = Image.fromarray(myImg)
    return myImg.resize((int(myImg.width/2), int(myImg.height/2)), Image.ANTIALIAS)

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
    show(img)

    plt.show()