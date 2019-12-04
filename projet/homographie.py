#!/usr/bin/env python3

from imgproc import openImage, show, resize
from dog import DoG
from keypoints import getKeyPoints, getOriginalCoordinates
from descriptor import descriptionPointsCles, assignOrientation
import numpy as np
import matplotlib.pyplot as plt

def distanceInterPoints(keypoints1, keypoints2):

    dists = np.empty((keypoints1.__len__(),keypoints2.__len__()), dtype=float)

    for k1 in range(0,keypoints1.__len__()):
        for k2 in range(0, keypoints2.__len__()):
            dists[k1][k2] = distanceEuclidienne(keypoints1[k1], keypoints2[k2])

    return dists

def distanceEuclidienne(p1,p2):
    dists = [];

    for i in range(2,p1.__len__()):
        dists.append(pow(p1[i] - p2[i],2))
       
    euclidianDist = sum(dists)

    return euclidianDist


if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    octave = 3
    scale = 4

    (diffs, imgs) = DoG(img, scale, octave)

    plt.plot([1,octave])

    descriptors = []

    for o in range(0,octave):
        survivants = []
        for s in range(0,scale):
            print(o, s)
            sigma = 2**(1/scale)
            survivants += getKeyPoints(diffs[o][s],diffs[o][s+1],diffs[o][s+2], sigma)

        keypoints = assignOrientation(diffs[o][s+1], survivants)
        octaveDescriptors = descriptionPointsCles(imgs[o][s+1], keypoints)
        descriptors.extend(octaveDescriptors)

        plt.subplot(1, octave, 1 + o)
        
        #show(imgs[o][0])
        show(openImage('Lenna.jpg'))

        x, y = [], []
        for (i,j,s,a,l) in keypoints:
            x.append(getOriginalCoordinates(i,o))
            y.append(getOriginalCoordinates(j,o))
            print("x : ", i, ", y : ", j, ", angle :", a, ", length : ", l)

        plt.title(o)
        plt.autoscale(False)
        plt.plot(x,y, 'bo', markersize=2)

    #plt.show()

    distanceInterPoints(descriptors, descriptors)
