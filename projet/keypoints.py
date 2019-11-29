#!/usr/bin/env python3

from imgproc import openImage, show, resize
from dog import DoG
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def localExtremaDetection(down, actual, up):
    if len(down) != len(actual) or len(actual) != len(up):
        return ([], [])

    maxima = []
    minima = []

    for y in range(1, len(actual)-1):
        for x in range(1, len(actual[y])-1):
            if isMaxima(down, actual, up, x, y):
                maxima.append((x,y))
            elif isMinima(down, actual, up, x, y):
                minima.append((x,y))
    return (maxima, minima)

def isMaxima(down, actual, up, y, x):
    target = actual[x][y]
    a = [actual[x-1][y-1], actual[x-1][y-0], actual[x-1][y+1],
         actual[x-0][y-1],                   actual[x-0][y+1],
         actual[x+1][y-1], actual[x+1][y-0], actual[x+1][y+1]]
    b = [0]#[down[x][y]]
    c = [0]#[up[x][y]]
    # b = [down[x-1][y-1], down[x-1][y-0], down[x-1][y+1],
    #      down[x-0][y-1], down[x-0][y-0], down[x-0][y+1],
    #      down[x+1][y-1], down[x+1][y-0], down[x+1][y+1]]
    # c = [up[x-1][y-1], up[x-1][y-0], up[x-1][y+1],
    #      up[x-0][y-1], up[x-0][y-0], up[x-0][y+1],
    #      up[x+1][y-1], up[x+1][y-0], up[x+1][y+1]]
    return max(a) <= target and max(b) <= target and max(c) <= target


def isMinima(down, actual, up, y, x):
    target = actual[x][y]
    a = [actual[x-1][y-1], actual[x-1][y-0], actual[x-1][y+1],
         actual[x-0][y-1],                   actual[x-0][y+1],
         actual[x+1][y-1], actual[x+1][y-0], actual[x+1][y+1]]
    b = [0]#[down[x][y]]
    c = [0]#[up[x][y]]
    # b = [down[x-1][y-1], down[x-1][y-0], down[x-1][y+1],
    #      down[x-0][y-1], down[x-0][y-0], down[x-0][y+1],
    #      down[x+1][y-1], down[x+1][y-0], down[x+1][y+1]]
    # c = [up[x-1][y-1], up[x-1][y-0], up[x-1][y+1],
    #      up[x-0][y-1], up[x-0][y-0], up[x-0][y+1],
    #      up[x+1][y-1], up[x+1][y-0], up[x+1][y+1]]
    return min(a) >= target and min(b) >= target and min(c) >= target

def contrastVerification(dog, candidates, limit): # limit = 0.03
    keypoints = []
    for candidate in candidates:
        if abs(dog[candidate[0]][candidate[1]]) < limit:
            keypoints.append(candidate)

    return keypoints

def eliminatingEdges(dog, candidates, limit): # limit = 10
    keypoints = []
    for candidate in candidates:
        dxx = dog[candidate[0]+1][candidate[1]]-2*dog[candidate[0]][candidate[1]]+dog[candidate[0]-1][candidate[1]]
        dxy = ((dog[candidate[0]+1][candidate[1]+1]-dog[candidate[0]-1][candidate[1]+1])-(dog[candidate[0]+1][candidate[1]-1]-dog[candidate[0]-1][candidate[1]-1]))/4
        dyy = dog[candidate[0]][candidate[1]+1]-2*dog[candidate[0]][candidate[1]]+dog[candidate[0]][candidate[1]-1]

        tr = dxx + dyy
        det = dxx*dyy - pow(dxy,2)

        if det > 0:
            ratio = pow(tr,2)/det
            threshold = pow(limit+1,2)/limit

            if ratio < threshold:
                keypoints.append(candidate)

    return keypoints


if __name__ == '__main__':

    img = openImage('Lenna.jpg')
    
    img = img / np.max(img)

    results = DoG(img, 4, 3)
    octave = 2
    scale = 1

    (maxima, minima) = localExtremaDetection(results[octave*4+scale-1], results[octave*4+scale], results[octave*4+scale+1])

    plt.plot([2,3])
    plt.subplot(231)
    show(results[octave*4+scale-1])
    plt.subplot(232)
    show(results[octave*4+scale])
    plt.subplot(233)
    show(results[octave*4+scale+1])
    plt.subplot(235)
    show(results[octave*4+scale])

    x, y = [], []
    for i,j in minima:
        x.append(i)
        y.append(j)

    plt.autoscale(False)
    plt.plot(x,y, 'bo', markersize=2)

    x, y = [], []
    for i,j in maxima:
        x.append(i)
        y.append(j)

    plt.autoscale(False)
    plt.plot(x,y, 'ro', markersize=2)

    plt.show()
