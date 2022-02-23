import numpy as np
import argparse
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # replace following functions with my functions


    #load the image
    imagehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define the list of boundaries
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
   
    #find colors within the specified boundaries and apply the mask
    mask = cv2.inRange(imagehsv, lower_blue, upper_blue)
    output = cv2.bitwise_and(imagehsv, imagehsv, mask=mask)

    #cleaning up image
    kernel = np.ones((15,15),np.uint8)
    opening = cv2.morphologyEx(output, cv2.MORPH_CLOSE,kernel)


    #hexagon location
    imageG = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    imageLL = cv2.threshold(imageG,50,255,cv2.THRESH_BINARY)
    tup=cv2.findNonZero(imageLL[1])
    tup2= cv2.mean(tup)
    print("Blue tape located at ", round(tup2[0]), round(tup2[1]))
    print(27*(((opening.shape[1])/2)-(tup2[0]))/(opening.shape[1]/2))


   
    # Display the resulting frame
    cv2.imshow('imageLL', opening)
    #cv2.imshow('frame', opening)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
