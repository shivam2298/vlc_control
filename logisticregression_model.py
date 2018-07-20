# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

path = 'C:\\Users\\Praveen Rustagi\\.spyder-py3\\vlc_control\\datasetimages\\'

d = []
y = []

for i in range(1,5):       #signs
    for j in range(300):    #images
        d.append(cv2.imread(path+str(i)+'\\'+str(j)+'.jpg'))
        y.append(i-1) #label as sign number


#preprocessing
pixel_data = np.array(d)
print (pixel_data.shape)
data = pixel_data.flatten().reshape(1200, 12288)
print (data.shape)
Y = np.array(y).reshape(1200,)

#print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.40)



#standard normalisation instead of mean and deviation one
X_train=X_train/255
X_test=X_test/255



clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

filename = 'saved_model.sav'
pickle.dump(clf, open(filename, 'wb'))
