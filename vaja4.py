from curses.ascii import ESC
import time
from cv2 import imshow, moments, normalize
from numba import jit
import cv2
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('media/usb.mp4')
fps = 1 / 30


# 1=rocna izbira meanshift 2 camshift
# 3=avtomatsko meanshift 4 camshift
MODE = 1

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
moment je % na katerem je mediana
49... je 50 torej je sredina
ce je 25 pa je isto kot minus 25% od centra

meanshift()
	for x:
		pridobimo histogram objekta
		ga normaliziramo
		projekcija ti najde piksle ki se ujemajo
		iz projekcije zracunas momente
		mement ti pove kam se okno premakne
"""


"""
karkol openCv ponuja

kuture - točke ki posisujejo seznam točk za vsak objekt

slika razlik - kaj se premika
ce se vse premika lahko z gavsovim šumom razresimo delno

označimo s klikom od kje bo sledilo

objekt želimo čiml bol natančno označit

KNN bolj stabilna

sliko flipaj da bo lažje

MINHIFT 
sledi objektu na osnovi barv
iz objekta ustvarimo model ki je histogram, prej jo pretvorimo v HSV
za računanje histograma cv2.calchist()

v vsakem frejmu uporabimo ta objekt da ga najdemo kje se nahaja
histogram je treba pred preverjanjem normalizirat - cv2.normalize()
povratna proekcija bo posvetlila vse piksle z istimi barvi cv2.callBackProject()
- preberemo sivino v sliki, v histogram gledamo kje se ta pixl nahaja ter vrnemo vrednost kje se nahaja

nato moramo najti težišče svetlih točk - to delamo z momenti Moo = vsota vseh sivin, M1o = vsota sivin*x ...
momenti omogočajo izračun težišč x= M10/Moo, y= Mo1/Moo    NA WIKIPEDIJI lazja formula? obstaja tudi cv knjiznica
to računanje moramo večkrat ponovit zaradi velikih premikov (for i=10 npr.)
škatlo premikamo proti težišču
ko se vse iteracije koncajo dobimo novi frejm in ponovimo vse

CAMSHIFT
razlika je da se velikost škatle lahko spreminja
spreminja se po formuli 1= W*sqr(Moo/256), H=1,2*W

igraj se da dobiš dobre vrednosti

HSV od 0 do 180

objekt ki mu sledimo z miško čimbolje označimo

zelo odvisna od maske ki jo iščemo
čimbolj učinkovito implementirat momente(vektor)

sami implemetirate minshift"""

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
    global  xLZ, yLZ, xLast, yLast

    xLast, yLast = (0,0)

    hist = None

    # izbira kvadrata
    while True:
        time.sleep(fps/2)
        ret, frame = cap.read()
        if ret:
            if xLZ != 0 or yLZ != 0:
                cv2.rectangle(frame,(xLZ,yLZ),(xLast,yLast),(0,255,0),2)
            
            cv2.imshow("frame", frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        ###

        if xLZ > xLast:
            xLZ, xLast = xLast, xLZ
        if yLZ > yLast:
            yLZ, yLast = yLast, yLZ

        frameHSV =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cutObject = frameHSV[yLZ:yLast,xLZ:xLast]
        # popravi ce v napacno stran oznacis
        #mask = cv2.inRange(frameRGB, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

        if xLZ != 0 or yLZ != 0:
            hist = cv2.calcHist([cutObject],[0],None,[256],[0,256])
            cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        
        if hist is not None:
            projekcija = cv2.calcBackProject([frameHSV],[0],hist,[0,256],1)
            cv2.normalize(projekcija,projekcija,0,255,cv2.NORM_MINMAX)
        
            imshow("Pro", projekcija)

        ###

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

def meanShift():
    global  xLZ, yLZ, xLast, yLast

    print(f"{xLZ},{yLZ},{xLast},{yLast}")

    # popravi ce izberes kvadratek v cudni smeri
    if xLZ > xLast:
        xLZ, xLast = xLast, xLZ
    if yLZ > yLast:
        yLZ, yLast = yLast, yLZ

    # if xLast - xLZ > yLast - yLZ:
    #     yLast +=  (xLast - xLZ) - (yLast - yLZ)

    # elif yLast - yLZ > xLast - xLZ:
    #     xLast += (yLast - yLZ) - (xLast - xLZ)

    print(f"{xLZ},{yLZ},{xLast},{yLast}")

    hist = None
    while True:
        ret, frame = cap.read()
        frameRGB =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cutObject = frameRGB[yLZ:yLast,xLZ:xLast]
        # popravi ce v napacno stran oznacis
        #mask = cv2.inRange(frameRGB, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

        
        if hist is None:
            hist = cv2.calcHist([cutObject],[0],None,[256],[0,256])
            cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        #cv2.imshow("cut", hist)

        projekcija = cv2.calcBackProject([frameRGB],[0],hist,[0,256],5)

        cv2.normalize(projekcija,projekcija,0,255,cv2.NORM_MINMAX)

        katic = projekcija[yLZ:yLast,xLZ:xLast]

        momenti = moments(katic)

        imshow("pro", projekcija[yLZ:yLast,xLZ:xLast])  
        imshow("proc", projekcija)      

        if momenti['m00'] == 0:
            momenti['m00'] += 1

        momentX = momenti['m10']/momenti['m00']
        momentY = momenti['m01']/momenti['m00']
        moveX = int((cutObject.shape[0]) * momentX/100 - (cutObject.shape[0]*0.5))
        moveY = int((cutObject.shape[1]) * momentY/100 - (cutObject.shape[1]*0.5))

        print(f"Moments {momentX} {momentY}")

        # nekak naredi da nemrejo zunaj skatle
        # if not (xLZ + moveX) < 0 | (xLZ + moveX) > frame.shape[0]:
        #     xLZ += moveX

        # if not (xLast + moveX) < 0 | (xLast + moveX) > frame.shape[0]:
        #     xLast += moveX

        # if not (yLZ + moveY) < 0 | (yLZ + moveY) > frame.shape[0]:
        #     yLZ += moveY

        # if not (yLast + moveY) < 0 | (yLast + moveY) > frame.shape[0]:
        #     yLast += moveY

        img2 = cv2.rectangle(frame, (xLZ,yLZ), (xLast,yLast), 255,2)

        cv2.imshow("frame", img2)

        c = cv2.waitKey(1)
        if c == ESC:
            exit()
    
def camShift():
    global  xLZ, yLZ, xLast, yLast
    return

if MODE in (1,2):
    selectObject()
elif MODE == (3,4):
    findObject()
else:
    print("Ne se spilat poba")
    exit()

if MODE in (1,3):
    meanShift()
elif MODE == (2,4):
    camShift()
else:
    print("Ne se spilat poba")
    exit()

cv2.destroyAllWindows()