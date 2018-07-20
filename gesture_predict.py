# -*- coding: utf-8 -*-


import cv2
import numpy as np
import requests
import pickle


def inc_volume():
    
    #print ('http://127.0.0.1:8080/requests/status.xml?command=volume&val=<100>')
    page = requests.get('http://127.0.0.1:8080/requests/status.xml?command=volume&val=+10')
    #print (page)

def dec_volume():
    
    #print ('http://127.0.0.1:8080/requests/status.xml?command=volume&val=<100>')
    page = requests.get('http://127.0.0.1:8080/requests/status.xml?command=volume&val=-10')
    #print (page)

def pause():
    
    #print ('http://127.0.0.1:8080/requests/status.xml?command=volume&val=<100>')
    page = requests.get('http://127.0.0.1:8080/requests/status.xml?command=pl_pause')
    #print (page)

def play():
    
    #print ('http://127.0.0.1:8080/requests/status.xml?command=volume&val=<100>')
    page = requests.get('http://127.0.0.1:8080/requests/status.xml?command=pl_play')
    #print (page)

loaded_model = pickle.load(open('saved_model.sav', 'rb'))
            
prev_cmd = -1
 
cap = cv2.VideoCapture(0)
    
i  =0
while(cap.isOpened()):
    # read image
    ret, img = cap.read()
    if img is not None:
        img = cv2.flip(img,1)
        #print(img.shape)
        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (300,120),(600,300), (0,255,0),2)
        crop_img = img[120:300, 300:600]
    
        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
        cv2.imshow('Original',img)
        cv2.imshow('greyscale',grey)
        res = cv2.resize(grey,(64, 64), interpolation = cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
        x_pred = rgb.flatten().reshape(1, 12288)
        player_cmd = loaded_model.predict(x_pred)[0]
        print(player_cmd)
        if not(player_cmd == prev_cmd) or (player_cmd==0 or player_cmd==1):
            if(player_cmd==0):
                inc_volume()
            elif player_cmd==1:
                dec_volume()
            elif player_cmd==2:
                play()
            elif player_cmd==3:
                pause()
            
            prev_cmd = player_cmd
                
            
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
        
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    