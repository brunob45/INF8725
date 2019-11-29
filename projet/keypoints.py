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

def isMaxima(down, actual, up, x, y):
    target = actual[y][x]
    a = [actual[y-1][x-1], actual[y-1][x-0], actual[y-1][x+1],
         actual[y-0][x-1],                   actual[y-0][x+1],
         actual[y+1][x-1], actual[y+1][x-0], actual[y+1][x+1]]
    # b = [0]#[down[y][x]]
    # c = [0]#[up[y][x]]
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]
    return max(a) <= target and max(b) <= target and max(c) <= target


def isMinima(down, actual, up, x, y):
    target = actual[y][x]
    a = [actual[y-1][x-1], actual[y-1][x-0], actual[y-1][x+1],
         actual[y-0][x-1],                   actual[y-0][x+1],
         actual[y+1][x-1], actual[y+1][x-0], actual[y+1][x+1]]
    # b = [0]#[down[y][x]]
    # c = [0]#[up[y][x]]
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]
    return min(a) >= target and min(b) >= target and min(c) >= target

def contrastVerification(dog, candidates, limit): # limit = 0.03
    keypoints = []
    for candidate in candidates:
        if abs(dog[candidate[1]][candidate[0]]) < limit:
            keypoints.append(candidate)

    return keypoints

def eliminatingEdges(dog, candidates, limit): # limit = 10
    keypoints = []
    for candidate in candidates:
        dxx = dog[candidate[1]+1][candidate[0]]-2*dog[candidate[1]][candidate[0]]+dog[candidate[1]-1][candidate[0]]
        dxy = ((dog[candidate[1]+1][candidate[0]+1]-dog[candidate[1]-1][candidate[0]+1])-(dog[candidate[1]+1][candidate[0]-1]-dog[candidate[1]-1][candidate[0]-1]))/4
        dyy = dog[candidate[1]][candidate[0]+1]-2*dog[candidate[1]][candidate[0]]+dog[candidate[1]][candidate[0]-1]

        tr = dxx + dyy
        det = dxx*dyy - pow(dxy,2)

        if det > 0:
            ratio = pow(tr,2)/det
            threshold = pow(limit+1,2)/limit

            if ratio < threshold:
                keypoints.append(candidate)

    return keypoints

def getKeyPoints(dog, s, o):
    img = dog[s + o * scale]
    # (maxima, minima) = localExtremaDetection(dog[s-1 + o * scale], img, dog[s+1 + o * scale])
    (maxima, minima) = localExtremaDetection(dog[s + o * scale], img, dog[s + o * scale])

    survivants = maxima + minima
    survivants = contrastVerification(img, survivants, limit=0.03)
    survivants = eliminatingEdges(img, survivants, limit=10)
    return survivants

if __name__ == '__main__':
    img = openImage('droite.jpg')

    octave = 3
    scale = 4

    results = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for o in range(0,octave):
        for s in range(1,scale-1):
            print(o, s)
            img = results[s + o * scale]
            survivants = getKeyPoints(results, s, o)

            plt.subplot(octave, scale, 1 + o*scale +s)
            show(img)

            x, y = [], []
            for i,j in survivants:
                x.append(i)
                y.append(j)

            plt.autoscale(False)
            plt.plot(x,y, 'bo', markersize=2)


    plt.show()
