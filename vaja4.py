from curses.ascii import ESC
from cv2 import BackgroundSubtractorMOG2, createBackgroundSubtractorMOG2
from numba import jit
import cv2
import time
from cv2 import cvtColor
import numpy as np
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('BigSmokeShort.mp4')

"""
detekcija gibanja/ starting point lahko vgrajeno metodo

iskanje verjetnosti uporabis povratno projekcijo kanala H

iskanje srednjega premika uporabis moment

mas link kako racunat

camshaft mors sam uporabit

torej zaznavanje gibanja mas opencv, sledenje objektu pa naredis sam
"""


SizeX = 320
SizeY = 240

cv2.namedWindow("frame")
cv2.namedWindow("mask")

prev = None
subtractor = createBackgroundSubtractorMOG2()


while True:
    # start = time.time()
    ret, frame = cap.read()
    if ret:
    
        resized = cv2.resize(frame, (SizeX, SizeY), interpolation=cv2.INTER_AREA)
        RGBframe = cvtColor(resized, cv2.COLOR_BGR2RGB)

        mask=subtractor.apply(frame)

        # cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

    c = cv2.waitKey(1)
    if c == ESC:
        break

    # while time.time() < (start + (1 / 30)):
    #     time.sleep(1/60)

cap.release()
cv2.destroyAllWindows()