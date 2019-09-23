# -*- coding: utf-8 -*-
"""TENSORFLOW + MOG.ipynb

**First of all change runtime to GPU**
"""

!wget  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

!tar -xvzf  faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

import os
if not os.path.exists(".../prince_dhruvin/AND PERSON DETECTION/silhouette_processed/"):
  os.makedirs(".../prince_dhruvin/MOG2 AND PERSON DETECTION/silhouette_processed/")

model_path = '/content/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.5

SILL_OUT_DIR = ".../prince_dhruvin/MOG2 AND PERSON DETECTION/silhouette_processed/"

cap = cv2.VideoCapture('.../prince_dhruvin/MOG2 AND PERSON DETECTION/10.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold = 0)
#fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold = 16)
count = 0
while True:
    r, frame = cap.read()
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (1280, 720))
    
    fgmask = fgbg.apply(frame)
    
    ret,thresh = cv2.threshold(fgmask,220,255,cv2.THRESH_BINARY)
    
    #thresh  = cv2.GaussianBlur(thresh, (5,5), 0)
    
    boxes, scores, classes, num = odapi.processFrame(frame)
    
    
    
    roi = []
    if classes[0] == 1 and scores[0] > threshold:
        count+=1
        #print("a")
        #print(boxes[0])
        (x, y, w, h) = boxes[0]
        roi = thresh[x : w, y  : h ]
        
        IMAGE_NAME = str(count)+".jpg"
        #cv2.imwrite(SILL_OUT_DIR+IMAGE_NAME, roi )
        
        #roi = cv2.imread(SILL_OUT_DIR+IMAGE_NAME, 0)
        #cv2.imshow("roi", roi)
        
        #count+=1
        
        _, contours, hierarchy = cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
        for i in range(len(contours)):
            #print("b")
        
            cnt = contours[i]
            
            area = cv2.contourArea(cnt)
            #print(area)
            if area > 2500:
                #print("c")
                #print(area)
                
                xx, yy, ww, hh = cv2.boundingRect(cnt)
               
                #print(xx, yy, ww, hh)
                #break
                crop = roi[yy : yy+ hh, xx : xx+ ww ]
                
                crop = image_resize(crop, height = 122)
                cv2.imwrite("of_no_use.jpg", crop)
                temp_image = Image.open("of_no_use.jpg", 'r')
                
                temp_image_w, temp_image_h = temp_image.size
                background = Image.new('RGB', (88, 122), (0, 0, 0))
                bg_w, bg_h = background.size
                offset = ((bg_w - temp_image_w) // 2, (bg_h - temp_image_h) // 2)
                background.paste(temp_image, offset)
                #IMAGE_NAME = str(count)+".jpg"
                #background = cv2.resize(background,(90, 120),interpolation=cv2.INTER_AREA)
                background.save(SILL_OUT_DIR+"sill"+IMAGE_NAME)
                #this = cv2.imread(SILL_OUT_DIR+im, 0)
                #this = cv2.resize(this,(90, 120),interpolation=cv2.INTER_AREA)
                #cv2.imwrite(SILL_OUT_DIR+im, this)
                
                
                #crop = image_resize(crop, height = 120)
                
            else: continue
                
            
        
        
        

    # Visualization of the results of a detection.

#     for i in range(len(boxes)):
#         # Class 1 represents human
#         if classes[i] == 1 and scores[i] > threshold:
#             box = boxes[i]
#             print(box)
#             cv2.rectangle(thresh,(box[1],box[0]),
#                               (box[3] + 10 ,box[2] + 10),(255,0,0),2)

#     cv2.imshow("preview", thresh)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

#following code is for zipping contents form gdrive to download it faster



output_filename = ".../prince_dhruvin/MOG2 AND PERSON DETECTION/07_with_ margin_silhouette_processed"
dir_name =  ".../prince_dhruvin/MOG2 AND PERSON DETECTION/silhouette_processed/"

import shutil
shutil.make_archive(output_filename, 'zip', dir_name)

import numpy as np
import cv2

cap = cv2.VideoCapture('IMG_2298.MOV')

fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 200)
fgbg2 = cv2.createBackgroundSubtractorKNN(dist2Threshold = 400)

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    frame = cv2.GaussianBlur(frame,(5,5),0)

    fgmask1 = fgbg1.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    
    thresh1 = fgmask1


    ret, thresh1 = cv2.threshold(thresh1, 252,255, cv2.THRESH_BINARY)
    #thresh1 = cv2.GaussianBlur(thresh1,(3,3),0)
    #ret, thresh1 = cv2.threshold(thresh1, 240,255, cv2.THRESH_BINARY)
    

    cv2.imshow('frame',thresh1)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
import cv2 as cv
from pylab import *
import numpy as np
from scipy.spatial import distance
import cv2
import pandas as pd
import pickle
from sklearn.externals import joblib


import os

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

##boy_path = 'gei/boy/'
##girl_path = 'gei/girl/'
# subject_path = 'gei/girl/g1.png'
angle_path = "/Users/prince_dhruvin/Desktop/Gender_ArbitraryView/CASIA B GEI/"
#lst_angle =  os.listdir(angle_path)
lst_angle_five = []
lst_angle_one = []
lst_gender_one = []
lst_gender_five = []
for gender in os.listdir(angle_path):
    gender = gender +'/'
    for person in  os.listdir(angle_path+gender):
        person = person + '/'
        for angles in  os.listdir(angle_path+gender+person): 
            #print(angle)
                angles = angles + '/' 
                #path_image.append(parent_dir+name+angles+angle)
                for gei in os.listdir(angle_path+gender+person+angles):
                    if angles[0:5] in [ "nm-01" ,"nm-02" ,"nm-03" , "nm-04" ]:
                        lst_gender_five.append(angle_path+gender+person+angles+gei)
                    elif angles[0:5] in ["nm-05", "nm-06"]:
                        lst_gender_one.append(angle_path+gender+person+angles+gei)
                    else: None
                    

data = list() 
data_one = list()
row_data = list()
row_data_one = list()


l = []
l_one = []


# list number of files
#sample_count = len(lst_angle) #ilosc probek
print('Data load started')


for gei in lst_gender_five:
    temp = gei.split('/')
    gender = "person - "+temp[7]
    #print(temp)
    #break
    #fn = angle_path+gei
    img =  cv.imread(gei, cv.IMREAD_GRAYSCALE)
    ar = np.asarray(img)
    #print(gei[-7: -4])
    # print(ar.shape)
    #gei looks like this 045_nm-02_144_GEI.png
    #id = gei[-7: -4]
    id = gender
    data.append((id, ar))
    #l.append(str(gei[-7: -4]))
    l.append(str(gender))
    row_data.append((id, ar.flatten()))
    # column_data.append((id, ar.flatten('F')))
    
#COLLECTING UNSEEN DATA FROM 06 
    
for gei in lst_gender_one:
    temp = gei.split('/')
    gender = "person - "+temp[7]
    #fn = angle_path+gei
    img =  cv.imread(gei, cv.IMREAD_GRAYSCALE)
    ar = np.asarray(img)
    #print(gei[-7: -4])
    # print(ar.shape)
    #gei looks like this 045_nm-02_144_GEI.png
    #id = gei[-7: -4]
    id = gender
    data_one.append((id, ar))
    #l_one.append(str(gei[-7: -4]))
    l_one.append(str(gender))
    
    row_data_one.append((id, ar.flatten()))

##for gei in lst2:
##    fn = girl_path+gei
##    ar = np.asarray(img)
##    # print(ar.shape)
##    id = gei
##    # print(id)
##    data.append((id, ar))
##    l.append("girl")
##    row_data.append((id, ar.flatten()))
##    # column_data.append((id, ar.flatten('F')))    
  
    

# print('Data load completed, no of samples {0}, of size {1}-{2}'.format(sample_count, ar.shape[0], ar.shape[1]))
print('Creating data matrix and performing pca')

X = np.vstack([item[1] for item in row_data])
y = np.vstack([item[0] for item in row_data])


X_UNSEEN = np.vstack([item[1] for item in row_data_one])
Y_UNSEEN = np.vstack([item[0] for item in row_data_one])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 55)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.99)

X_train = pca.fit_transform(X_train)
#pca1 = PCA(n_components = X_train.shape[1] )
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import f1_score,accuracy_score, classification_report
print(classification_report(y_test, y_pred))
z = f1_score(y_test, y_pred, average='weighted') 
print("F1-score: ",z)


parent_dir = "/Users/prince_dhruvin/Desktop/Gender_ArbitraryView/GEI_MODEL/GEI_120x90/"


save_model_path = "/Users/prince_dhruvin/Desktop/Gender_ArbitraryView/GEI_MODEL/GEI_120x90/"
filename = 'arbitary_person_logistic.sav'
joblib.dump(classifier, save_model_path+filename)

pickle.dump(pca, open( save_model_path+"pca_person.p", "wb" ) )
pickle.dump(sc, open( save_model_path+"sc_person.s", "wb" ) )
np.save(save_model_path+'X_UNSEEN_PERSON',X_UNSEEN)
np.save(save_model_path+'Y_UNSEEN_PERSON',Y_UNSEEN)