import cv2
from cv2 import imshow
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def gudMoments(image, x:int, y:int):
    value = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value+=image[i][j]*pow(i,x)*pow(j,y)
    return value

def momentTest():
    xLZ, yLZ = (0, 0)
    inputImage = cv2.imread('media/momentTest.png', cv2.IMREAD_COLOR)
    inputImage = cv2.flip(inputImage, 1)
    xLast, yLast = (inputImage.shape[0], inputImage.shape[1])


    #imshow("input", inputImage)

    inputImage = inputImage[:,:,2]

    momenti = cv2.moments(inputImage)

    print(gudMoments(inputImage, 0, 0))
    print(gudMoments(inputImage, 0, 1))
    print(gudMoments(inputImage, 1, 0))

    print(momenti)



    moveX = momenti['m10']/momenti['m00']
    moveY = momenti['m01']/momenti['m00']

    print(momenti['m00'])
    print(momenti['m01'])
    print(momenti['m10'])

    print (f"Premik x: {moveX}")
    print (f"Premik y: {moveY}")

    inputImage[int(moveX)][int(moveY)] = 120

    #imshow("test", inputImage)

    cv2.waitKey(0)

def histTest():
    inputImage = cv2.imread('media/momentTest.png', cv2.IMREAD_COLOR)

    frameRGB =  cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([frameRGB],[1],None,[256],[0,256])
    
    plt.hist(hist, bins = 256)
    plt.show()

histTest()