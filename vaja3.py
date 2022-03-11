from ast import Constant
from curses.ascii import ESC
from math import sqrt
from cv2 import INTER_MAX
from numba import jit
import cv2
import numpy as np

# Stevilo iteracij
NumIterations = 1

inputImage = cv2.imread('landscape.jpg', cv2.IMREAD_COLOR)

def euclidianDistance(point1, point2):
    distance = sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
    return distance

def bestFit(pixel, options:list):
    print(options)
    bestDistance = 999999
    best = 0
    for i in range(len(options)):
        distance = euclidianDistance(options[i], pixel)
        if(distance < bestDistance):
            bestDistance = distance
            best = i

    return best

#@jit(nopython=True)
def pasterilize(image, numK:int):
    # range mam od 0 do 255 za vsako koordinato
    # mediane enakomerno porazdelim
    # 
    medians=[]
    for i in range(1,numK+1):
        value = i * (255/(numK+1))
        medians.append([value, value, value])

    medianPixels=[[]] * numK
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            print(f"Best fit is {bestFit(image[x][y], medians)}")
            (medianPixels[1]).append(3)
            #medianPixels[bestFit(image[x][y], medians)].append(image[x][y])
            print(medianPixels)
            exit()
            #median = medianAverage()
    return image

#ustvarim kopijo
outputImage = inputImage[:]
for i in range(1):
    outputImage = pasterilize(outputImage, 2)

exit()

cv2.namedWindow("Image")

c = '0'
while c != ESC:
    cv2.imshow("Original", np.hstack([outputImage]))
    c = cv2.waitKey(500)

cv2.destroyAllWindows()
