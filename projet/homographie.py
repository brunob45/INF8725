#!/usr/bin/env python3

from imgproc import openImage, show, resize
from dog import differenceDeGaussiennes
from keypoints import detectionPointsCles, getOriginalCoordinates
from descriptor import descriptionPointsCles, assignOrientation
import numpy as np
import matplotlib.pyplot as plt

def distanceInterPoints(keypoints1, keypoints2):

    dists = np.empty((keypoints1.__len__(),keypoints2.__len__()), dtype=float)
    print("Range : ", keypoints1.__len__()*keypoints2.__len__())
    for k1 in range(0,keypoints1.__len__()):
        for k2 in range(0, keypoints2.__len__()):
            dists[k1][k2] = distanceEuclidienne(keypoints1[k1], keypoints2[k2])
            print(dists[k1][k2], " : ", k1*k2, " out of ",keypoints1.__len__()*keypoints2.__len__())

    return dists

def distanceEuclidienne(p1,p2):
    dists = [];

    for i in range(2,p1.__len__()):
        dists.append(pow(p1[i] - p2[i],2))
       
    euclidianDist = sum(dists)

    return euclidianDist

def getDescriptorsImage(name):
    img = openImage(name)

    octave = 3
    scale = 4

    (diffs, imgs) = differenceDeGaussiennes(img, scale, octave)

    descriptors = []

    for o in range(0,octave):
        survivants = detectionPointsCles(diffs[o], seuil_contraste=0.02)

        keypoints = assignOrientation(diffs[o][0], survivants)
        octaveDescriptors = descriptionPointsCles(imgs[o][0], keypoints)
        descriptors.extend(octaveDescriptors)

    return descriptors

if __name__ == '__main__':
    descDroite = getDescriptorsImage('droite.jpg')
    descGauche = getDescriptorsImage('gauche.jpg')

    mat = distanceInterPoints(descDroite, descGauche)

    np.save("couples.npy", mat)
