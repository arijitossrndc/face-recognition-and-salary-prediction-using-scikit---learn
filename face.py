import cv2
import numpy as np
import time
import keyboard
from tkinter import *



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

'''global file

def pop():
    def return_c():
        global file
        name.get()
        window.destroy()

    window=Tk()
    window.wm_title('oss')
    
    l1=Label(window,text="Name : ")
    
    l1.grid(row=0,column=0)
    name=StringVar()
    e1=Entry(window,textvariable=name)
    e1.grid(row=0,column=1)
    b1=Button(window,text="Save",width=18,command=return_c)
    b1.grid(row=1,column=0)
    window.mainloop()'''


while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,
        minNeighbors=5)

    eyes = eyeCascade.detectMultiScale(gray,scaleFactor=1.05,
        minNeighbors=5)


    #smiles = smileCascade.detectMultiScale(gray,scaleFactor=1.05,
        #minNeighbors=10)

    for (xf, yf, wf, hf) in faces:
        cv2.rectangle(frame, (xf, yf), (xf+wf, yf+hf), (0, 255, 0), 2)

    for (xe, ye, we, he) in eyes:
        cv2.rectangle(frame, (xe, ye), (xe+we, ye+he), (255, 0, 0), 2)

    #for (xs, ys, ws, hs) in smiles:
        #cv2.rectangle(frame, (xs, ys), (xs+ws, ys+hs), (0, 0, 255), 2)


    cv2.imshow('Video', frame)

    cv2.waitKey(1)
    if keyboard.is_pressed('escape'):
        break
    '''if keyboard.is_pressed("c"):
        global file
        file =  pop()
        cv2.imwrite(file,frame)'''



video_capture.release()
cv2.destroyAllWindows()
