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

def contrastVerification(img, candidates, limit=0.03): # limit = 0.03
    keypoints = []

    print("Total candidates : " + candidates.__len__().__str__())
    for candidate in candidates:
        dx = (img[candidate[1]][candidate[0]+1]-img[candidate[1]][candidate[0]-1])/2
        dy = (img[candidate[1]+1][candidate[0]]-img[candidate[1]-1][candidate[0]])/2

        if abs(dx) > limit or abs(dy) > limit:
            keypoints.append(candidate)
    print("Eliminated candidates by contrast : " + (candidates.__len__() - keypoints.__len__()).__str__())
    return keypoints

def eliminatingEdges(img, candidates, limit=10): # limit = 10
    print("Total candidates : " + candidates.__len__().__str__())
    keypoints = []
    for candidate in candidates:
        #dxx = img[candidate[0]+1][candidate[1]]-2*img[candidate[0]][candidate[1]]+img[candidate[0]-1][candidate[1]]
        #dxy = ((img[candidate[0]+1][candidate[1]+1]-img[candidate[0]-1][candidate[1]+1])-(img[candidate[0]+1][candidate[1]-1]-img[candidate[0]-1][candidate[1]-1]))/4
        #dyy = img[candidate[0]][candidate[1]+1]-2*img[candidate[0]][candidate[1]]+img[candidate[0]][candidate[1]-1]

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
    print("Eliminated candidates because on an edge : " + (candidates.__len__() - keypoints.__len__()).__str__())
    return keypoints

#def getPoint(img,x,y,octave):
    #return img[x/pow(2,octave)][y/pow(2,octave)]

def getKeyPoints(down,dog,up, s, o):
    #img = dog[s + o * scale]
    # (maxima, minima) = localExtremaDetection(dog[s-1 + o * scale], img, dog[s+1 + o * scale])
    #(maxima, minima) = localExtremaDetection(dog[s + o * scale], img, dog[s + o * scale])
    (maxima, minima) = localExtremaDetection(down, dog, up, s)

    survivants = maxima + minima
    survivants = contrastVerification(dog, survivants,0.03)
    survivants = eliminatingEdges(dog, survivants)
    return survivants

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    octave = 4
    scale = 6

    results,imgs = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for o in range(0,octave):
        for s in range(1,scale-1):
            print(o, s)
            img = results[s + o * scale]
            survivants = getKeyPoints(results[s-1 + o * scale],results[s + o * scale],results[s+1 + o * scale], s, o)

            plt.subplot(octave, scale, 1 + o*scale +s)
            show(img)

            x, y = [], []
            for i,j,s in survivants:
                x.append(i)
                y.append(j)

            plt.autoscale(False)
            plt.plot(x,y, 'bo', markersize=2)

    
    plt.show()
