from curses.ascii import ESC
from numba import jit
import cv2
import time
from cv2 import cvtColor
import numpy as np
# pygame, qt, tkinter, numba!!!!
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('BigSmokeShort.mp4')

PRECISION = 6

BoxX = 64
BoxY = 48
SizeX = 320
SizeY = 240

# lows = np.array([50, 5, 5]
# highs = np.array([100, 80, 50])

lows = np.array([130, 90, 70])
highs = np.array([180, 135, 125])


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

@jit(nopython=True)
def checkBlock(frame, x:int, y:int):
    count = 0
    for x1 in range(x, x+BoxY):
        for y1 in range(y, y+BoxX):
            [h,s,v] = frame[x1,y1]
            if h > lows[0] and h < highs[0] and s > lows[1] and s < highs[1] and v > lows[2] and v < highs[2]:
                count+=1
    return count

@jit(nopython=True)
def find_face(frame):
    maxCount=0
    faceX = 0
    faceY = 0

    for x in range(0,SizeY, BoxY / PRECISION):
        for y in range(0, SizeX, BoxX / PRECISION):

            count = checkBlock(frame, x, y)
            
            if count > maxCount:
                maxCount = count
                faceX = x
                faceY = y

    return (faceY, faceX)

# najdi diplomo profecorja in skopiraj

cv2.namedWindow("mask")
cv2.namedWindow("frame")

while True:
    start = time.time()
    ret, frame = cap.read()
    if ret:
    
        resized = cv2.resize(frame, (SizeX, SizeY), interpolation=cv2.INTER_AREA)
        RGBframe = cvtColor(resized, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Input', frame)
        # loop over the boundaries
        #print(RGBframe)

        #print("got img")
        x, y = find_face(RGBframe)
        #print("found face")
        #print(f"X: {x}, Y: {y}")

        mask = cv2.inRange(RGBframe, lows, highs)
        #output = cv2.bitwise_and(resized, resized, mask = mask)

        cv2.rectangle(mask, (x, y), (x+BoxX, y+BoxY), (255,20, 0),2)
        cv2.rectangle(resized, (x, y), (x+BoxX, y+BoxY), (255,20, 0),2)
        
        cv2.imshow("mask", mask)
        cv2.imshow("frame", resized)

    c = cv2.waitKey(1)
    if c == ESC:
        break

    while time.time() < (start + (1 / 30)):
        time.sleep(1/60)

cap.release()
cv2.destroyAllWindows()