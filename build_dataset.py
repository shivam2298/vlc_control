import cv2
import numpy as np
import math
import os

path = 'C:\\Users\\Praveen Rustagi\\.spyder-py3\\vlc_control\\datasetimages\\4\\'


cap = cv2.VideoCapture(0)
i = 0
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
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

       
    # show image
        cv2.imshow('Original',img)
        cv2.imshow('greyscale',crop_img)
        res = cv2.resize(gray,(64, 64), interpolation = cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
        print (rgb.shape)
        cv2.imwrite(os.path.join(path , str(i)+'.jpg'), rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if(i==299):
        break
    i+=1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    
    









