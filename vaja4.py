from curses.ascii import ESC
from cv2 import BackgroundSubtractorMOG2, createBackgroundSubtractorMOG2
from numba import jit
import cv2
import time
from cv2 import cvtColor
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('BigSmokeShort.mp4', cv2.FFMPEG)

# 1=rocna izbira, 2=avtomatsko
MODE = 2

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

SizeX = 320
SizeY = 240

global xLZ
global yLZ
global xLast
global yLast
xLZ = 0
yLZ = 0

# click event, posodobi vrednosti ko vleces
def clickOnImage(event, x, y, flags, param):
    global  xLZ, yLZ, xLast, yLast
    if event == cv2.EVENT_LBUTTONUP:
        #print("EVENT_LBUTTONUP ({},{})".format(x,y))
        xLZ = 0
        yLZ = 0
        
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("EVENT_LBUTTONDOWN ({},{})".format(x,y))
        xLZ = x
        yLZ = y
        
    if event == cv2.EVENT_MOUSEMOVE:
        #print("EVENT_MOUSEMOVE ({},{})".format(x,y))
        xLast = x
        yLast = y

def selectObject():
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", clickOnImage)

    # izbira kvadrata
    while True:
        # start = time.time()
        ret, frame = cap.read()
        if ret:
            if xLZ != 0 or yLZ != 0:
                cv2.rectangle(frame,(xLZ,yLZ),(xLast,yLast),(0,255,0),2)
            
            cv2.imshow("frame", frame)

        c = cv2.waitKey(1)
        if c == ESC:
            exit()
        if c == ord('r'):
            #uporabnik je zadovoljen z izbranim
            break

        # while time.time() < (start + (1 / 30)):
        #     time.sleep(1/60)

    cv2.destroyAllWindows()

def findObject():
    cv2.namedWindow("frame")
    global xLZ, yLZ, xLast, yLast
    backFrame = None

    # izbira kvadrata
    while True:
        ret, frame = cap.read()
        if ret:
            frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameG = cv2.GaussianBlur(frameG, (21, 21), 0)
            
            if backFrame is None:
                backFrame = frameG
                continue

        diff = cv2.absdiff(backFrame,frameG)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("Diff", diff)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:    
            c_max = max(contours, key=cv2.contourArea)
            xLZ,yLZ,w,h = cv2.boundingRect(c_max)
            xLast = xLZ + w
            yLast = yLZ + h
            cv2.rectangle(frame,(xLZ,yLZ),(xLast,yLast),(0,255,0),2)
            
            #narise obrobe
            #cv2.drawContours(frame, contours, -1 , (255, 0, 0), 1)

        cv2.imshow("frame", frame)

        c = cv2.waitKey(1)
        if c == ESC:
            exit()
        if c == ord('r'):
            return



    cv2.destroyAllWindows()

if MODE == 1:
    selectObject()
elif MODE == 2:
    findObject()
else:
    print("Ne se spilat poba")
    exit()

cv2.destroyAllWindows()