import pandas as pd
import numpy as np
import cv2
import os

data_root = '/home/dashmoment/dataset/ai_challenger_scene/validation'

ann_val_path = os.path.join(data_root, 'scene_validation_annotations_20170908.json')
ann_val = pd.read_json(ann_val_path)

img = cv2.imread(os.path.join(data_root, 'images', ann_val['image_id'][0]))

corp_ratio = 4.0/5.0
step = 1.0/5.0

x_step = int(img.shape[0]*step) -1
y_step = int(img.shape[1]*step) - 1

x_init = 0
x_end = int(img.shape[0]*corp_ratio)

img = cv2.flip(img,1)

while x_end < img.shape[0]:
    
    y_init = 0
    y_end = int(img.shape[1]*corp_ratio)
    
    while y_end < img.shape[1]:
        
        
        print(x_init, x_end,y_init, y_end)
        y_init = y_init + y_step
        y_end = y_end + y_step
        
    x_init = x_init + x_step
    x_end = x_end + x_step


center_corp = img[img.shape[0]//2 -int(img.shape[0]*corp_ratio)//2: img.shape[0]//2 +int(img.shape[0]*corp_ratio)//2, img.shape[1]//2 -int(img.shape[1]*corp_ratio)//2: img.shape[1]//2 +int(img.shape[1]*corp_ratio)//2]
cv2.imshow('origin', img)
cv2.imshow('cen', center_corp)

cv2.waitKey()

#for i in range(0,img.shape[0], img.shape[0]//step):
#    for j in range(0,img.shape[1], img.shape[1]//step):
#        print(i,j)

