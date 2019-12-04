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
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]

    maxima = max(a+b+c)
    maxa = max(a)
    maxb = max(b)
    maxc = max(c)

    if maxima < target:
        return True
    else:
        return False


def isMinima(down, actual, up, x, y):
    target = actual[y][x]
    a = [actual[y-1][x-1], actual[y-1][x-0], actual[y-1][x+1],
         actual[y-0][x-1],                   actual[y-0][x+1],
         actual[y+1][x-1], actual[y+1][x-0], actual[y+1][x+1]]
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]

    minima = min(a+b+c)
    mina = min(a)
    minb = min(b)
    minc = min(c)

    if minima > target:
        return True
    else:
        return False

def contrastVerification(img, candidates, limit=0.03):
    keypoints = []
    for (x,y) in candidates:
        dx = (img[y][x+1]-img[y][x-1])/2
        dy = (img[y+1][x]-img[y-1][x])/2

        if abs(dx) > limit or abs(dy) > limit:
            keypoints.append((x,y))
    print("Eliminated candidates by contrast:", len(candidates) - len(keypoints))
    return keypoints

def eliminatingEdges(img, candidates, limit=10):
    keypoints = []
    for (x,y) in candidates:
        dyy = img[y+1][x] - 2*img[y][x] + img[y-1][x]
        dxy = ((img[y+1][x+1] - img[y+1][x-1]) - (img[y-1][x+1] - img[y-1][x-1]))/4
        dxx = img[y][x+1] - 2*img[y][x] + img[y][x-1]

        tr = dxx + dyy
        det = dxx*dyy - pow(dxy,2)

        if det > 0:
            ratio = pow(tr,2)/det
            threshold = pow(limit+1,2)/limit

            if ratio < threshold:
                keypoints.append((x,y))

    print("Eliminated candidates because on an edge:", len(candidates) - len(keypoints))
    return keypoints

def getKeyPoints(down,dog,up, sigma):
    (maxima, minima) = localExtremaDetection(down, dog, up)

    survivants = maxima + minima
    print("Total candidates:", len(survivants))
    survivants = contrastVerification(dog, survivants)
    survivants = eliminatingEdges(dog, survivants)
    print("Surviving candidates:", len(survivants))

    result = []
    for (x,y) in survivants:
        result.append((x, y, sigma))
    return result

def getOriginalCoordinates(c,o):
    return c*pow(2,o)

if __name__ == '__main__':
    img = openImage('droite.jpg')

    octave = 2
    scale = 3

    (diffs, imgs) = DoG(img, scale, octave)

    plt.plot([1,octave])

    for o in range(0,octave):
        survivants = []
        for s in range(0,scale):
            print(o, s)
            sigma = 2**(1/scale)
            survivants += getKeyPoints(diffs[o][s],diffs[o][s+1],diffs[o][s+2], sigma)

        plt.subplot(1, octave, 1 + o)
        show(imgs[o][0])

        x, y = [], []
        for (i, j, k) in survivants:
            x.append(i)
            y.append(j)

        plt.title(o)
        plt.autoscale(False)
        plt.plot(x,y, 'bo', markersize=2)

    plt.show()