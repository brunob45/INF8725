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

def localExtremaDetection(down, actual, up, s):
    if len(down) != len(actual) or len(actual) != len(up):
        return ([], [])

    maxima = []
    minima = []

    for y in range(1, len(actual)-1):
        for x in range(1, len(actual[y])-1):
            if isMaxima(down, actual, up, x, y):
                maxima.append((x,y,s))
            elif isMinima(down, actual, up, x, y):
                minima.append((x,y,s))
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
    for candidate in candidates:
        dx = (img[candidate[1]][candidate[0]+1]-img[candidate[1]][candidate[0]-1])/2
        dy = (img[candidate[1]+1][candidate[0]]-img[candidate[1]-1][candidate[0]])/2

        if abs(dx) > limit or abs(dy) > limit:
            keypoints.append(candidate)
    print("Eliminated candidates by contrast:", len(candidates) - len(keypoints))
    return keypoints

def eliminatingEdges(img, candidates, limit=10):
    keypoints = []
    for candidate in candidates:
        dyy = img[candidate[1]+1][candidate[0]]-2*img[candidate[1]][candidate[0]]+img[candidate[1]-1][candidate[0]]
        dxy = ((img[candidate[1]+1][candidate[0]+1] - img[candidate[1]+1][candidate[0]-1]) - (img[candidate[1]-1][candidate[0]+1] - img[candidate[1]-1][candidate[0]-1]))/4
        dxx = img[candidate[1]][candidate[0]+1]-2*img[candidate[1]][candidate[0]]+img[candidate[1]][candidate[0]-1]


        tr = dxx + dyy
        det = dxx*dyy - pow(dxy,2)

        if det > 0:
            ratio = pow(tr,2)/det
            threshold = pow(limit+1,2)/limit

            if ratio < threshold:
                keypoints.append(candidate)
    print("Eliminated candidates because on an edge:", len(candidates) - len(keypoints))
    return keypoints

def getKeyPoints(down,dog,up, s, o):
    (maxima, minima) = localExtremaDetection(down, dog, up, s)

    survivants = maxima + minima
    print("Total candidates:", len(survivants))
    survivants = contrastVerification(dog, survivants)
    survivants = eliminatingEdges(dog, survivants)
    print("Surviving candidates:", len(survivants))
    return survivants

def getOriginalCoordinates(c,o):
    return c*pow(2,o)

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    octave = 4
    scale = 6

    results = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for o in range(0,octave):
        for s in range(1,scale-1):
            print(o, s)
            img = results[s + o * scale]
            survivants = getKeyPoints(results[s-1 + o * scale],results[s + o * scale],results[s+1 + o * scale], s, o)

            plt.subplot(octave, scale, 1 + o*scale +s)
            show(openImage('Lenna.jpg'))

            x, y = [], []
            for i,j,s in survivants:
                x.append(getOriginalCoordinates(i,o))
                y.append(getOriginalCoordinates(j,o))

            plt.autoscale(False)
            plt.plot(x,y, 'bo', markersize=2)

    
    plt.show()
