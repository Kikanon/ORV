from curses.ascii import ESC
from math import sqrt
from numba import jit
import cv2
import numpy as np

# 1. Kje se pojavijo razlike pri detekciji robov nad temno in svetlo sliko in zakaj?
# Ob veliki gostoti enakih barv (npr. črna) se izgubijo detajli oz. lokacija roba


# 2. Zakaj je pred uporabo detektorja robov smiselno uporabiti filter za glajenje?
# S tem se izognemo šumu, ki ga sami ne smatramo za robove

# 3. Kaj nam pove gradient slike? Kako se uporablja pri detektorjih robov?
# Gradient nam pove smer in intenziteto spreminjanja barve.
# Robovi imajo ponavadi visok gredient, ki kaze proti njim


dark = cv2.imread('media/burgirDark.png', cv2.IMREAD_GRAYSCALE)# burgirDark.png
light = cv2.imread('media/burgirLight.png', cv2.IMREAD_GRAYSCALE)

dark = cv2.GaussianBlur(dark, [3, 3], cv2.BORDER_DEFAULT)
light = cv2.GaussianBlur(light, [3, 3], cv2.BORDER_DEFAULT)

@jit(nopython=True)
def combineMasks(mask1, mask2):
    maskR = np.zeros(shape=mask1.shape)
    for x in range(mask1.shape[0]):
        for y in range(mask1.shape[1]):
            maskR[x][y]=sqrt(pow(mask1[x][y], 2) + pow(mask2[x][y], 2))
    return maskR

@jit(nopython=True)
def calculateMask(image, mask, _x, _y):
    # maska mora bit 3x3, pa center je pac vedno -1 -1
    sum = 0
    for x in range(0, 3):
        for y in range(0, 3):
            
            try:
                pixel=image[_x+x-1][_y+y-1]
            except:
                pixel=0

            sum += pixel * mask[x][y]
    #ce je out of bounds je pixel 0
    return int(abs(sum)/9) #(9*9*4)

@jit(nopython=True)
def Sobel (image):
    result1 = np.zeros(shape=image.shape)
    result2 = np.zeros(shape=image.shape)
    mask = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    mask2 = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result1[x][y]=calculateMask(image, mask, x, y)

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result2[x][y]=calculateMask(image, mask2, x, y)

    return combineMasks(result1, result2).astype(np.uint8)

@jit(nopython=True)
def Prewitt(image):
    result1 = np.zeros(shape=image.shape)
    result2 = np.zeros(shape=image.shape)
    mask = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    mask2 = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result1[x][y]=calculateMask(image, mask, x, y)

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result2[x][y]=calculateMask(image, mask2, x, y)

    return combineMasks(result1, result2).astype(np.uint8)

@jit(nopython=True)
def Roberts(image):
    result1 = np.zeros(shape=image.shape)
    result2 = np.zeros(shape=image.shape)
    mask = np.array([[0,0,0], [1,0,0], [0,-1,0]])
    mask2 = np.array([[0,0,0], [0,1,0], [-1,0,0]])

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result1[x][y]=calculateMask(image, mask, x, y)

    for x in range(0, image.shape[0], 1):
        for y in range(0, image.shape[1], 1):
            result2[x][y]=calculateMask(image, mask2, x, y)

    return combineMasks(result1, result2).astype(np.uint8)

cv2.namedWindow("Original")
cv2.namedWindow("Prewitt")
cv2.namedWindow("Sobel")
cv2.namedWindow("Roberto")
cv2.namedWindow("Canny")

cv2.imshow("Original", np.hstack([dark, light]))
cv2.imshow("Prewitt", np.hstack([Prewitt(light), Prewitt(dark)]) )
cv2.imshow("Sobel", np.hstack([Sobel(light), Sobel(dark)]) )
cv2.imshow("Roberto", np.hstack([Roberts(light), Roberts(dark)]) )
cv2.imshow("Canny", np.hstack([cv2.Canny(light, threshold1=40, threshold2=100), cv2.Canny(dark, threshold1=40, threshold2=100)]) )

c = '0'
while c != ESC:
    c = cv2.waitKey()

cv2.destroyAllWindows()
