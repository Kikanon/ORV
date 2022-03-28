from curses.ascii import ESC
from cv2 import BackgroundSubtractorMOG2, createBackgroundSubtractorMOG2
from numba import jit
import cv2
import time
from cv2 import cvtColor
import numpy as np
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('BigSmokeShort.mp4')

# zaznavanje lahko skopiras od Mlakara
# lahko naredis da dvakrat kliknes da definiras skatlo
# dobro met cim manjso skatlo
# uporabi flip da lazje testiras z kamero
# Camshift sledi na osnovi barve
# Ustvari neki model -> histogram za objekt ki mu sledimo
# Hostpgram -> H iz HSV ali R is RGB
# historgram(slika O)
# cv2.calchist() lahko uporabis
# cv2.normalize za histogram (ima vrednosti 0-1)
# cv2.calcbackproject ti da vse piksle, ki so podobni histogramu
# (vzame sivino, pogleda v histogramu in izpise vrednost)
# Moment je vsota vseh sivin
# 10 je vsota kjer se pomnozis z x
# y momenti dobimo centre nekih gruc
# aja, racunas moment te projekcije kje je in ti povejo novo tezisce
# racunanje momentov se zgodi iterativno (for)
# 10x je ok, moment je primik koordinat
# pri momentih racunas samo pixle znotraj skatlice
# to je meanshift
# camshift pa ma se en korak v for zanki(velikost skatle se lahko spreminja)
# sqrt(M00/256) je sirina
# visina je 1,2xsirina
# na zacetku cakas pa kliknes enter ko si zadovoln

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