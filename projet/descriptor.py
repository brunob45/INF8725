from imgproc import openImage, show, resize
from dog import differenceDeGaussiennes
from keypoints import detectionPointsCles, getOriginalCoordinates
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def descriptionPointsCles(img, keypoints):
    descriptors = []

    for keypoint in keypoints:
        descriptors.append(createDescriptor(img, keypoint))

    return descriptors

def createDescriptor(img, keypoint):
    descriptor = []

    # filtre gaussien
    x = np.linspace(-keypoint[2]*1.5, keypoint[2]*1.5, 17)
    d1 = np.diff(norm.cdf(x))
    d2 = np.outer(d1, d1)
    gaussianFilter = (d2/d2.sum())

    # extraction de la patch, rotation
    patch = rotate(keypoint[3], getPatch(img,keypoint[0],keypoint[1], 16))
    gradPatch = np.empty((16,16), dtype=object)

    # Calcul du gradient et application du point gaussien
    for i in range(0,16):
        for j in range(0,16):
            a,l = gradient(patch,i,j)
            w = l*gaussianFilter[i][j]
            gradPatch[i][j] = (a,w)

    # Division en 4x4 sous-régions
    subregions = []

    for region in range(0,4):
        startx = 0
        starty = 4
        subregion = np.empty((4,4), dtype=object)

        for i in range(startx,startx+4):
            for j in range(starty,starty+4):
                subregion[i - startx][j - starty] = gradPatch[i][j]

        if startx != 16:
            startx += 4
        else:
           if starty != 16:
               starty += 4
               startx = 0


        subregions.append(subregion)

    # Création des histogrammes
    hists = []

    for region in range(0,4):
        hist = np.zeros(8, dtype=np.float)
        for i in range(0,4):
            for j in range(0,4):
                a,l = subregions[region][i][j]
                test = int(np.floor(359)//(360//8))
                angle = int(np.floor(a)//(360//8))
                if angle == 8:
                    angle = 7
                hist[angle] += l
        hists.append(hist)

    # Construction du descripteur
    descriptor.append(keypoint[0]) # x
    descriptor.append(keypoint[1]) # y

    for hist in range(0,hists.__len__()):
        for value in range(0, 8):
            descriptor.append(hists[hist][value])

    return descriptor

def getPatch(img, x,y, size):
    patch = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            if x-int(np.floor(size/2))+i < img.shape[0] and x-int(np.floor(size/2))+i > 0 and y-int(np.floor(size/2))+j < img.shape[1] and y-int(np.floor(size/2))+j > 0:
                patch[i][j] = img[x-int(np.floor(size/2))+i][y-int(np.floor(size/2))+j]

    return patch

def rotate(angle, mat):
    #rotMat = np.array((np.cos(np.radians(angle)), -np.sin(np.radians(angle)), np.sin(np.radians(angle)), np.cos(np.radians(angle))))
    #return rotMat.dot(mat)
    return mat

def assignOrientation(img, keypoints):
    orientedKeypoints = []

    for keypoint in keypoints:
        width = int(2*np.ceil(keypoint[2]*1.5)+1)
        x = np.linspace(-keypoint[2]*1.5, keypoint[2]*1.5, 2*width+2)
        d1 = np.diff(norm.cdf(x))
        d2 = np.outer(d1, d1)
        gaussianFilter = (d2/d2.sum())
        hist = np.zeros(36, dtype=np.float)

        for j in range(-width, width+1):
            for i in range(-width, width+1):
                if keypoint[0]+i < 0 or keypoint[0]+i > img.shape[0]-1: # x
                    continue
                if keypoint[1]+j < 0 or keypoint[1]+j > img.shape[1]-1: # y
                    continue
                angle, length = gradient(img,keypoint[0]+i,keypoint[1]+j)
                a = int(np.floor(angle)//(360//36))
                w = gaussianFilter[j+width, i+width]*length
                hist[a] += w

        # index du max dans hist
        indexMax = 0;
        for i in range(0,hist.__len__()):
            if hist[i] > hist[indexMax]:
                indexMax = i

        maxValue = hist[indexMax] # length
        xmax = fitParabola(hist, indexMax) # angle

        orientedKeypoints.append((keypoint[0], keypoint[1], keypoint[2], xmax, maxValue))

    return orientedKeypoints

def gradient(img,x,y): # img is the corresponding smoothed image
    dy = img[min(y+1,img.shape[1]-1),x] - img[max(y-1,0),x]
    dx = img[y,min(x+1,img.shape[0]-1)] - img[y,max(x-1,0)]

    angleRad = np.arctan2(dy,dx)
    anglePol = (angleRad+np.pi)*180.0/np.pi
    length = np.sqrt(pow(dx,2)+pow(dy,2))

    return anglePol, length

def fitParabola(hist, indexMax): #10
    if indexMax == 0:
        x1 = -5
        y1 = hist[0]
    else:
        x1 = (indexMax - 1)*10+5
        y1 = hist[indexMax - 1]
    x2 = indexMax*10+5
    y2 = hist[indexMax]
    if indexMax == hist.__len__()-1:
        x3 = hist.__len__()*10+5
        y3 = hist[hist.__len__()-1]
    else:
        x3 = (indexMax + 1)*10+5
        y3 = hist[indexMax + 1]

    return (x2 + 0.5*((y1-y2)*pow((x3-x2),2)-(y3-y2)*pow((x2-x1),2))/((y1-y2)*(x3-x2)+(y3-y2)*(x2-x1)))

def get_descriptors(img, s, nb_octave):
    descriptors = []

    (diffs, imgs) = differenceDeGaussiennes(img, s, nb_octave)

    for octave in range(0,nb_octave):
        survivants = detectionPointsCles(diffs[octave], seuil_contraste=0.02)

        keypoints = assignOrientation(diffs[octave][0], survivants)
        octaveDescriptors = descriptionPointsCles(imgs[octave][0], keypoints)
        descriptors.extend(octaveDescriptors)

        plt.subplot(1, nb_octave, 1 + octave)

    return descriptors


if __name__ == '__main__':

    img = openImage('Lenna.jpg')

    octave = 3
    scale = 4

    (diffs, imgs) = differenceDeGaussiennes(img, scale, octave)

    descriptors = []

    for o in range(0,octave):
        survivants = detectionPointsCles(diffs[o], seuil_contraste=0.02)

        keypoints = assignOrientation(diffs[o][0], survivants)
        # octaveDescriptors = descriptionPointsCles(imgs[o][0], keypoints)
        # descriptors.extend(octaveDescriptors)

        output = np.empty((4,0))
        for (i,j,s,a,l) in keypoints:
            np.append(output, [i,j,s,a])

    np.save("keypoints.npy", output)
