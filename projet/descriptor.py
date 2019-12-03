from imgproc import openImage, show, resize
from dog import DoG
from keypoints import getKeyPoints
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def descriptionPointsCles(img, keypoints):
    descriptors = []

    for keypoint in keypoints:
        pass


    return descriptors

def createDescriptor(img, keypoint):
    descriptor = []
    descriptor.append[keypoint[0]]
    descriptor.append[keypoint[1]]




    return descriptor

def assignOrientation(img, keypoints):
    orientedKeypoints = []

    for keypoint in keypoints:
        width = int(2*np.ceil(keypoint[2]*1.5)+1)
        #gaussian = gaussian_filter(np.ones((2*width+1,2*width+1)),keypoint[2]*1.5)
        x = np.linspace(-keypoint[2]*1.5, keypoint[2]*1.5, 2*width+2)
        #x = np.linspace(-2.5, 2.5, 5+1)
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

        #for # bin

        orientedKeypoints.append((keypoint[0], keypoint[1], keypoint[2], angle, length))

    return orientedKeypoints

def gradient(img,x,y): # img is the corresponding smoothed image
    dy = img[min(y+1,img.shape[1]-1),x] - img[max(y-1,0),x]
    dx = img[y,min(x+1,img.shape[0]-1)] - img[y,max(x-1,0)]

    angleRad = np.arctan2(dy,dx)
    anglePol = (angleRad+np.pi)*180.0/np.pi
    length = np.sqrt(pow(dx,2)+pow(dy,2))

    return anglePol, length

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    octave = 3
    scale = 4

    results,imgs = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for o in range(0,octave):
        for s in range(1,scale-1):
            print(o, s)
            img = results[s + o * scale]
            survivants = getKeyPoints(results[s-1 + o * scale],results[s + o * scale],results[s+1 + o * scale], s, o)
            keypoints = assignOrientation(imgs[s + o * scale], survivants)

            plt.subplot(octave, scale, 1 + o*scale+s)
            show(img) # show scalled image instead of original

            x, y = [], []
            for i,j,s in keypoints:
                #if s > 1:
                    #x.append(i*pow(2,o*scale+s-1))
                    #y.append(j*pow(2,o*scale+s-1))
                #else:
                x.append(i)
                y.append(j)

            plt.autoscale(False)
            plt.plot(x,y, 'bo', markersize=2)

    
    plt.show()
