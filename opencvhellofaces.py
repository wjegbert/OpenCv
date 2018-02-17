# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:40:11 2017

@author: William
"""
import numpy as np
import cv2

capture = cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier(
        'C:\\Users\\William\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    # Our operations on the frame come here
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_detect.detectMultiScale(gray, 1.3, 5)
    #draw box around face? 
      
        for (x,y,w,h) in faces:
            print (str(x)+" "+str(y) + " "+ str(w) + " " +str(h))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            for (ex, ey, ew, eh) in roi_color:
                

                
            

    
    #detect face?
  
        
    # Display the resulting frame
    cv2.imshow('frame',frame)

    # Closes all windows when 1 or q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()