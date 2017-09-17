import pandas as pd
import numpy as np
import cv2
import os
import random
import config

conf = config.config("home")

def calculate_data_attribute(data_sets, data_set_files, file_path = None):
    
    img_means = []
    total_pixel = []
    axis0_size = 0
    axis1_size = 0
    
    total_Nfile = 0
    
    for dset, ann in zip(data_sets, data_set_files):  
        
        for i in range(len(ann)):
            
            total_Nfile = total_Nfile + 1
            
            print("DataSet: {}, progress:{}/{}".format(dset, i, len(ann)))
            
            
            img = cv2.imread(os.path.join(dset, 'images',ann['image_id'][i]))
            
            img_means.append([img[:,:,0].sum(), img[:,:,1].sum(), img[:,:,2].sum()]) 
            total_pixel.append(img.shape[0]*img.shape[1])
            axis0_size = axis0_size + img.shape[0]
            axis1_size = axis1_size + img.shape[1]
    
    img_means = np.vstack(img_means)
    total_pixel = np.sum(total_pixel)
    means = np.sum(img_means,axis=0)/total_pixel
    
    avg_axis0 = axis0_size/total_Nfile
    avg_axis1 = axis1_size/total_Nfile
    
    attributes = {
            "Nfiles":total_Nfile,
            "avg_axis0":avg_axis0,
            "avg_axis1":avg_axis1,
            "avg_mean":means,
    
            }

    if file_path != None:
        
        df = pd.DataFrame(attributes)
        df.to_csv(file_path, index=False)

    return attributes

#data_sets = [training_set]
#data_set_files = [ann_train]
#file_path =  os.path.join(data_root,'attributes.csv')
#d_attr = calculate_data_attribute(data_sets, data_set_files,file_path)