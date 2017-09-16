import pandas as pd
import numpy as np
import cv2
import os
import random

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

def random_crop(img, corp_ratio = 0.8, step = 0.2):

    portions = []
    init_point = 0
    
    
    while init_point < 1- corp_ratio + 0.1*step:
        
        portions.append(init_point)
        
        init_point = init_point + step
    
    
    random.shuffle(portions) 
    x_init = int(portions[0]*img.shape[0])
    x_end= int(x_init + corp_ratio*img.shape[0])
    random.shuffle(portions) 
    y_init = int(portions[0]*img.shape[1])
    y_end = int(y_init + corp_ratio*img.shape[1])
    
    crop_img = img[x_init:x_end,y_init:y_end,:]
    
    return crop_img

def random_flip(img):
    
    flip_code = [0,1,2]
    random.shuffle(flip_code) 
     
    if flip_code[0] != 2:
        flip_img = cv2.flip(img, flip_code[0])
    
    else:
        flip_img = img
    
    return flip_img



def get_batch(data_root, ann_file, batch_size, index_list, step ,image_size=(224,224), isflip = True):

    batch_img = []
    batch_label = []
    
    for idx in range(step*batch_size,step*batch_size + batch_size):
        
        i = index_list[idx]
        img = cv2.imread(os.path.join(data_root, 'images',ann_file['image_id'][i]))
        
        if img == None:
            print("No such file ", os.path.join(data_root, 'images',ann_file['image_id'][i]))
            return
        
        label = ann_file['label_id'][i]
        tmp_img = random_crop(img)
        if isflip == True: tmp_img = random_flip(tmp_img)   
        tmp_img = cv2.resize(tmp_img, image_size)
        
        batch_img.append(tmp_img)
        batch_label.append(label)
        
    batch_img = np.stack(batch_img)
    
    return batch_img, batch_label
    
