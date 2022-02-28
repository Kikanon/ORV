from curses.ascii import ESC
import cv2
from cv2 import cvtColor
import numpy
# pygame, qt, tkinter, numba!!!!

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# color ranages that we will use
boundaries = [
    ([120, 78, 64], [186,142,125]),
    ([151, 141, 160], [246, 227, 237])
	#([0, 0, 40], [110, 100, 200]),
]

# najdi diplomo profecorja in skopiraj
while True:
    ret, frame = cap.read()
    #frame = cvtColor(frame, cv2.COLOR_RGB2HSV)
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    #cv2.imshow('Input', frame)

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = numpy.array(lower, dtype = "uint8")
        upper = numpy.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mak2 = frame.copy()
        mak2[1:100,1:100,2] = 255
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask = mask)
        # show the images
        cv2.imshow("images", numpy.hstack([frame, output, mak2]))

    c = cv2.waitKey(1)
    if c == ESC:
        break

cap.release()
cv2.destroyAllWindows()