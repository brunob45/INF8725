#!/usr/bin/env python3

from imgproc import openImage, show, resize
from dog import differenceDeGaussiennes
from keypoints import detectionPointsCles, getOriginalCoordinates
from descriptor import descriptionPointsCles, assignOrientation
import numpy as np
import matplotlib.pyplot as plt

def distanceInterPoints(keypoints1, keypoints2):

    dists = np.empty((keypoints1.__len__(),keypoints2.__len__()), dtype=float)

    # traverse tout les keypoints de l'image 1 et 2 et calcul leur distance euclidienne
    for k1 in range(0,keypoints1.__len__()):
        for k2 in range(0, keypoints2.__len__()):
            dists[k1][k2] = distanceEuclidienne(keypoints1[k1], keypoints2[k2])

    return dists

def distanceEuclidienne(p1,p2):
    dists = [];

    # calcul des des différences aux carrés
    for i in range(2,p1.__len__()):
        dists.append(pow(p1[i] - p2[i],2))
       
    # calcul de la racine carré de la somme
    euclidianDist = np.sqrt(sum(dists))

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

    # code de test pour sauvegarde la matrice distances.npy (le calcul est très long à faire à chaque fois sinon)
    #mat = distanceInterPoints(descDroite, descGauche)
    #np.save("distances.npy", mat)

    distMat = np.load("./distances.npy")

    keypointsMin1 = []
    keypointsMin2 = []

    for i in range(0,10):
        ks = np.unravel_index(distMat.argmin(), distMat.shape)
        k1 = ks[0]
        k2 = ks[1]
        distMat[k1][k2] += 9999999
        keypointsMin1.append(descDroite[k1])
        keypointsMin2.append(descGauche[k2])

    plt.subplot(122)
    show(openImage('droite.jpg'))

    x = []
    y = []
    for k1 in keypointsMin1:
        x.append(k1[0])
        y.append(k1[1])
    plt.plot(x, y, marker='o', linewidth=0, markersize=5)

    plt.subplot(121)
    show(openImage('gauche.jpg'))

    x = []
    y = []
    for k2 in keypointsMin2:
        x.append(k2[0])
        y.append(k2[1])
    plt.plot(x, y, marker='o', linewidth=0, markersize=5)
    plt.show()
