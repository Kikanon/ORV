from curses.ascii import ESC
from re import S
from sqlite3 import Time
from tabnanny import check
from telnetlib import NOP
from cv2 import FastFeatureDetector_NONMAX_SUPPRESSION
from numba import jit
import cv2
from cv2 import cvtColor
import numpy as np
# pygame, qt, tkinter, numba!!!!
cap = cv2.VideoCapture(0)

# color ranages that we will use
colors = [
    #([120, 78, 64], [186,142,125]),
    #([151, 141, 160], [246, 227, 237]),
    #([85,63,72], [140,129,134]),
	#([176,169,187], [219,227,244])
    ([0,0,100], [94,6,42])
]

upperSkin = np.array([9, 100, 230])
lowerSkin = np.array([0, 40, 120])

BoxX = 64
BoxY = 48
SizeX = 320
SizeY = 240

lowH = 0
highH = 94

lowS = 2
highS = 50

lowV = 42
highV = 100

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


@jit(nopython=True)
def checkBlock(frame, x:int, y:int):
    count = 0
    for x in range(x, x+BoxX):
        for y in range(y, y+BoxY):
            h = frame[x][y][0]
            s = frame[x][y][2]
            v = frame[x][y][3]
            if h > lowH & h < highH & s > lowS & s < highS & v > lowV & v < highV:
                count+=1
    return count

@jit(nopython=True)
def find_face(frame):
    maxCount=0
    faceX = 0
    faceY = 0

    for x in range(0,SizeX-BoxX, BoxX):
        for y in range(0, SizeY-BoxY, BoxY):

            count = checkBlock(frame, x, y)
            
            if count > maxCount:
                maxCount = count
                faceX = x
                faceY = y

    #         >>> ball = img[280:340, 330:390]
    #         >>> img[273:333, 100:160] = ball
    return (faceX, faceY)

# najdi diplomo profecorja in skopiraj

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    HSVframe = cvtColor(resized, cv2.COLOR_BGR2HSV)
    #cv2.imshow('Input', frame)
    # loop over the boundaries

    x, y = find_face(HSVframe)

    #print(f"X: {x}, Y: {y}")

    cv2.rectangle(resized, (x, y), (x+BoxX, y+BoxY), (255,20, 0),2)

    mask = cv2.inRange(resized, (lowH, lowS, lowV), (highH, highS, highV))
    output = cv2.bitwise_and(resized, resized, mask = mask)

    cv2.namedWindow("slikice")
    cv2.imshow("slikice", np.hstack([resized, HSVframe, output]))

    c = cv2.waitKey(1)
    if c == ESC:
        break

cap.release()
cv2.destroyAllWindows()