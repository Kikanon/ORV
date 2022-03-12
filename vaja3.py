from ast import Constant
from curses.ascii import ESC
from math import sqrt
from cv2 import INTER_MAX
from numba import jit
import cv2
import numpy as np

# Stevilo iteracij
numIterations = 5
# Stevilo skupin
numK = 5

inputImage = cv2.imread('media/landscape.jpg', cv2.IMREAD_COLOR)

@jit(nopython=True)
def euclidianDistance(point1, point2):
    distance = sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
    return distance

@jit(nopython=True)
def bestFit(pixel, options):
    bestDistance = 999999
    best = 0
    for i in range(len(options)):
        distance = euclidianDistance(options[i], pixel)
        if(distance < bestDistance):
            bestDistance = distance
            best = i

    return best

#@jit(nopython=True)
def updateMedians(image, medians):
    medianSums = np.array([np.array([0,0,0]) for _ in range(numK)])
    medianCounts = np.array([0]*numK)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            selected = bestFit(image[x][y], medians)
            medianSums[selected] = np.array([medianSums[selected][0]+image[x][y][0],medianSums[selected][1]+image[x][y][1],medianSums[selected][2]+image[x][y][2]])
            medianCounts[selected] += 1
    
    for i in range(len(medians)):
        medians[i] = np.divide(medianSums[i], medianCounts[i])
            
#@jit(forceobj=True)
def calcMedians(image):
    # zacetne tocke
    medians=np.array([[0,0,0] for _ in range(numK)])
    for i in range(1,numK+1):
        value = i * (255/(numK+1))
        medians[(i-1)]=np.asarray([value, value, value])

    # racunanje median
    for i in range(numIterations):
        updateMedians(image, medians)
    
    return medians

def reduceImage(image, colors):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x][y] = colors[bestFit(image[x][y], colors)]
    return image

#ustvarim kopijo
outputImage = np.copy(inputImage)

medians = calcMedians(outputImage)

outputImage = reduceImage(outputImage, medians)

first_color = ( int (medians[0] [ 0 ] ), int (medians[0] [ 1 ] ), int (medians[0] [ 2 ] ))
colors = np.hstack([cv2.rectangle(np.zeros((10, 10, 3), np.uint8), (0,0),(9,9), first_color, -1)])

for i in range(1,len(medians)):
    square = ( int (medians[i] [ 0 ] ), int (medians[i] [ 1 ] ), int (medians[i] [ 2 ] ))
    colors = np.hstack([
        colors,
        cv2.rectangle(np.zeros((10, 10, 3), np.uint8), (0,0),(9,9), square, -1)
    ])

cv2.namedWindow("Image")
cv2.namedWindow("Pasterized")
cv2.namedWindow("Original")

c = '0'
while c != ESC:
    cv2.imshow("Colors", colors)
    cv2.imshow("Pasterized", outputImage)
    cv2.imshow("Original", inputImage)
    c = cv2.waitKey(500)

cv2.destroyAllWindows()
