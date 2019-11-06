#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image

def openImage(filename):
    return Image.open(filename).convert('L')

def show(img):
    plt.imshow(img, cmap='gray')#, vmin=0, vmax=255)

def resize(img):
    myImg = Image.fromarray(img)
    return myImg.resize( (int(myImg.width/2), int(myImg.height/2)) , Image.ANTIALIAS)

if __name__ == '__main__':
    img = openImage('droite.jpg')
    show(img)
    plt.show()