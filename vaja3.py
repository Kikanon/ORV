from curses.ascii import ESC
from math import sqrt
from numba import jit
import time
import cv2
import numpy as np

# Stevilo iteracij
numIterations = 5
# Stevilo skupin
numK = 255
# Velikost kvadratkov v paleti
paletteBoxSize=30

inputImage = cv2.imread('media/plaza.jpg', cv2.IMREAD_COLOR)

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

@jit(nopython=True)
def sumPixels(image, sums, counts, indexes):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            selected = bestFit(image[x][y], indexes)
            sums[selected] = np.array([sums[selected][0]+image[x][y][0],sums[selected][1]+image[x][y][1],sums[selected][2]+image[x][y][2]])
            counts[selected] += 1

def updateMedians(image, medians):
    # ustvarim arraye
    medianSums = np.array([np.array([0,0,0]) for _ in range(numK)])
    medianCounts = np.array([0]*numK)
    
    # sestejem vse pixle
    sumPixels(image, medianSums, medianCounts, medians)
    
    # nastavim mediane na nove vrednosti
    for i in range(len(medians)):
        medians[i] = np.divide(medianSums[i], medianCounts[i])
            
def calcMedians(image):

    # zacetne tocke
    #medians=[[0,0,0], [20,20,20], [255,255,255]]
    medians=np.array([[0,0,0] for _ in range(numK)])
    for i in range(1,numK+1):
        value = i * (255/(numK+1))
        medians[(i-1)]=np.asarray([value, value, value])

    # racunanje median
    for i in range(numIterations):
        updateMedians(image, medians)
    
    return medians

@jit(nopython=True)
def reduceImage(image, colors):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x][y] = colors[bestFit(image[x][y], colors)]
    return image

def makePalet(colors):
    if len(colors) < 1:
        return []
    first_color = ( int (colors[0] [ 0 ] ), int (colors[0] [ 1 ] ), int (colors[0] [ 2 ] ))
    pallet = np.hstack([cv2.rectangle(np.zeros((paletteBoxSize, paletteBoxSize, 3), np.uint8), (0,0),(paletteBoxSize-1,paletteBoxSize-1), first_color, -1)])
    for i in range(1,len(colors)):
        square = (  int (colors[i][0]),
                    int (colors[i][1]),
                    int (colors[i][2]))
        pallet = np.hstack([
            pallet,
            cv2.rectangle(np.zeros((paletteBoxSize, paletteBoxSize, 3), np.uint8), (0,0),(paletteBoxSize-1,paletteBoxSize-1), square, -1)
        ])

    return pallet

if __name__ == "__main__":
    start_time = time.time()

    #ustvarim kopijo
    outputImage = np.copy(inputImage)

    # zracunam mediane
    medians = calcMedians(outputImage)

    # barve prilagodim medianam
    outputImage = reduceImage(outputImage, medians)

    # sestavim barvne kvadratke za prikaz uporabljenih barv
    colors = makePalet(medians)

    print(f"Generated in {round(time.time()-start_time, 2)}s")

    cv2.namedWindow("Colors")
    cv2.namedWindow("Pasterized")
    cv2.namedWindow("Original")

    c = '0'
    while c != ESC:
        cv2.imshow("Colors", colors)
        cv2.imshow("Pasterized", outputImage)
        cv2.imshow("Original", inputImage)
        c = cv2.waitKey(500)

    cv2.destroyAllWindows()
