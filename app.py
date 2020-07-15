import cv2, time
import pandas as pd
import datetime

video = cv2.VideoCapture(0)
c = 1

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    c = c + 1
    check,frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=5)

    for x,y,w,h in face:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('Frame',frame)
    

    key = cv2.waitKey(1)    

    if key == ord('q'):
        break


print("Total Frames : ",c)
video.release()
cv2.destroyAllWindows()
