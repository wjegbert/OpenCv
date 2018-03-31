# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:40:11 2017

@author: William
"""
import numpy as np
import cv2
#captures feed from default webcam
capture = cv2.VideoCapture(0)
#initializes built-in face detection algo based on Haar cascades
face_detect=cv2.CascadeClassifier(
        '...opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    # Our operations on the frame come here
    if ret is True:
        #convert to greyscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #implement built-in face detection algorithm 
        faces=face_detect.detectMultiScale(gray, 1.3, 5)
   
        #draw box around face
        for (x,y,w,h) in faces:
            #x,y,w,h are outputs from the face detection classifier 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #creates region of interest around face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w] 
  
        
    # Display the resulting frame with face in box
    cv2.imshow('frame',frame)

    # ends loop when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
