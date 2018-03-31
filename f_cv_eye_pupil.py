# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:40:11 2017

@author: William
"""
import numpy as np
import cv2
import scipy.signal as sig
capture = cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier(
        'C:\\Users\\Student\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier(
        'C:\\Users\\Student\\Anaconda3\\pkgs\\opencv3-3.1.0-py35_0\\Library\\etc\\haarcascades\\haarcascade_eye.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    # Our operations on the frame come here
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_detect.detectMultiScale(gray, 1.3, 5)
    #draw box around face? 
      
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_detect.detectMultiScale(roi_color, 1.3, 5)
            eye = eyes[0:2]
            for (ex, ey, ew, eh) in eye:
                 xi = ex + 10
                 yi = ey + 10
                 wi = ew - 20
                 hi = eh - 20
                 cv2.rectangle(roi_color,(xi, yi),(xi + wi, yi + hi),(0, 255,0),2)
                 roeye_gray = roi_gray [ yi: yi + hi, xi: xi + wi]
                 roeye_color = roi_color[yi: yi + hi, xi: xi + wi]
                 darkspots = sig.argrelmin(roeye_gray)
                 pup = np.mean(darkspots, dtype=int)
                 cv2.circle(frame, (pup+xi+x, pup+yi+y), 5, (0, 0, 255), 2)
                 
                 
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # Closes all windows when 1 or q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()