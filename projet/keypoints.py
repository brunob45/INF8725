#!/usr/bin/env python3

from imgproc import openImage, show, resize
from dog import differenceDeGaussiennes

import numpy as np
import matplotlib.pyplot as plt


def localExtremaDetection(down, actual, up):
    # safety verification
    if len(down) != len(actual) or len(actual) != len(up):
        return ([])

    # find maximums and minimums
    extrema = []
    for y in range(1, len(actual)-1):
        for x in range(1, len(actual[y])-1):
            if isMaxima(down, actual, up, x, y):
                extrema.append((x,y))
            elif isMinima(down, actual, up, x, y):
                extrema.append((x,y))
    return extrema

def isMaxima(down, actual, up, x, y):
    # center point
    target = actual[y][x]

    # all the neighbors around the center point in the same scale
    a = [actual[y-1][x-1], actual[y-1][x-0], actual[y-1][x+1],
         actual[y-0][x-1],                   actual[y-0][x+1],
         actual[y+1][x-1], actual[y+1][x-0], actual[y+1][x+1]]
    # all the neighbors around the center point in the scale below
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    # all the neighbors around the center point in the scale above
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]

    # all the neighbors are regrouped
    maxima = max(a+b+c)

    # determine if the center point is greater than its neigbors
    if maxima < target:
        return True
    else:
        return False


def isMinima(down, actual, up, x, y):
    # center point
    target = actual[y][x]

    # all the neighbors around the center point in the same scale
    a = [actual[y-1][x-1], actual[y-1][x-0], actual[y-1][x+1],
         actual[y-0][x-1],                   actual[y-0][x+1],
         actual[y+1][x-1], actual[y+1][x-0], actual[y+1][x+1]]
    # all the neighbors around the center point in the scale below
    b = [down[y-1][x-1], down[y-1][x-0], down[y-1][x+1],
         down[y-0][x-1], down[y-0][x-0], down[y-0][x+1],
         down[y+1][x-1], down[y+1][x-0], down[y+1][x+1]]
    # all the neighbors around the center point in the scale above
    c = [up[y-1][x-1], up[y-1][x-0], up[y-1][x+1],
         up[y-0][x-1], up[y-0][x-0], up[y-0][x+1],
         up[y+1][x-1], up[y+1][x-0], up[y+1][x+1]]

    # all the neighbors are regrouped
    minima = min(a+b+c)

    # determine if the center point is smaller than its neigbors
    if minima > target:
        return True
    else:
        return False

def contrastVerification(img, candidates, limit=0.03):
    keypoints = []

    # check the difference of value of the pixel with its direct neighbors
    for (x,y) in candidates:
        dx = (img[y][x+1]-img[y][x-1])/2
        dy = (img[y+1][x]-img[y-1][x])/2
        
        # verify if the difference in value (contrast) is big enough to keep it
        if abs(dx) > limit or abs(dy) > limit:
            keypoints.append((x,y))
    print("Eliminated candidates by contrast:", len(candidates) - len(keypoints))
    return keypoints

def eliminatingEdges(img, candidates, limit=10):
    keypoints = []

    # calculate the hessian matrix values
    for (x,y) in candidates:
        dyy = img[y+1][x] - 2*img[y][x] + img[y-1][x]
        dxy = ((img[y+1][x+1] - img[y+1][x-1]) - (img[y-1][x+1] - img[y-1][x-1]))/4
        dxx = img[y][x+1] - 2*img[y][x] + img[y][x-1]

        # calculate the trace and determinant of the hessian matrix
        tr = dxx + dyy
        det = dxx*dyy - pow(dxy,2)

        # if the determinant is inferior to zero, we do not keep it
        if det > 0:
            ratio = pow(tr,2)/det
            threshold = pow(limit+1,2)/limit
            # check if curvature ratio is big enough to be a corner
            if ratio < threshold:
                keypoints.append((x,y))

    print("Eliminated candidates because on an edge:", len(candidates) - len(keypoints))
    return keypoints

def getOriginalCoordinates(c,o):
    # return the coordinates of a downscale image in the original
    return c*pow(2,o)

def detectionPointsCles(DoG, sigma=1.0, seuil_contraste=0.03, r_courb_principale=10, resolution_octave=3):
    survivants = []
    for s in range(2, len(DoG)):
        img = DoG[s-1]
        kps = localExtremaDetection(DoG[s-2],DoG[s-1],DoG[s])

        print("Total candidates:", len(kps))
        kps = contrastVerification(img, kps, seuil_contraste)
        kps = eliminatingEdges(img, kps, r_courb_principale)
        print("Surviving candidates:", len(kps))

        for (x,y) in kps:
            survivants.append((x, y, sigma))

    return survivants


if __name__ == '__main__':
    img = openImage('droite.jpg')

    octave = 3
    scale = 4

    (diffs, imgs) = differenceDeGaussiennes(img, scale, octave)

    ax = plt.subplot(111)
    plt.title('droite')
    show(img)
    plt.autoscale(False)

    for o in range(0,octave):
        survivants = detectionPointsCles(diffs[o], seuil_contraste=0.02)

        x, y = [], []
        for (i, j, k) in survivants:
            x.append(getOriginalCoordinates(i,o))
            y.append(getOriginalCoordinates(j,o))

        ax.plot(x, y, marker='o', linewidth=0, markersize=5, label=str(o))


    ax.legend()

    plt.show()