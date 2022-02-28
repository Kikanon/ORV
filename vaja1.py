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
    #([120, 78, 64], [186,142,125]),
    #([151, 141, 160], [246, 227, 237]),
    #([85,63,72], [140,129,134]),
	#([176,169,187], [219,227,244])
    ([150,100,100],[255,255,255])
]

# najdi diplomo profecorja in skopiraj
while True:
    ret, frame = cap.read()
    cvtColor(frame, cv2.COLOR_RGB2HSV_FULL)
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    #cv2.imshow('Input', frame)

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = numpy.array(lower, dtype = "uint8")
        upper = numpy.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        box = cv2.rectangle(frame, (10,10), (20, 20), (20,20, 20),2)
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask = mask)
        # show the images
        cv2.namedWindow("slikice")
        cv2.imshow("slikice", numpy.hstack([frame, output, box]))

    c = cv2.waitKey(1)
    if c == ESC:
        break

cap.release()
cv2.destroyAllWindows()