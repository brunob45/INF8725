from imgproc import openImage, show, resize
from dog import DoG
from keypoints import getKeyPoints, getOriginalCoordinates
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

if __name__ == '__main__':
    img = openImage('Lenna.jpg')

    octave = 3
    scale = 4

    results = DoG(img, scale, octave)

    plt.plot([octave,scale])

    for o in range(0,octave):
        for s in range(1,scale-1):
            print(o, s)
            img = results[s + o * scale]
            survivants = getKeyPoints(results[s-1 + o * scale],results[s + o * scale],results[s+1 + o * scale], s, o)
            keypoints = assignOrientation(results[s + o * scale], survivants)

            plt.subplot(octave, scale, 1 + o*scale+s)
            show(openImage('Lenna.jpg'))

            x, y = [], []
            for i,j,s,a,l in keypoints:                
                x.append(getOriginalCoordinates(i,o))
                y.append(getOriginalCoordinates(j,o))
                print("x : ", i, ", y : ", j, ", angle :", a, ", length : ", l)

            plt.autoscale(False)
            plt.plot(x,y, 'bo', markersize=2)

    
    plt.show()
