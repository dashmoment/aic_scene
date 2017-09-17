import pandas as pd
import numpy as np
import cv2
import os
import random
import config


class data_utility: 
    def __init__(self, congfiguration, corp_ratio = 0.8, corp_step = 0.2):

        self.conf = config.config(congfiguration)
        self.corp_ratio = corp_ratio
        self.corp_step = corp_step

    def random_crop(self,img):

        portions = []
        init_point = 0
        
        
        while init_point < 1- self.corp_ratio + 0.1*self.corp_step:
            
            portions.append(init_point)
            
            init_point = init_point + self.corp_step
        
        
        random.shuffle(portions) 
        x_init = int(portions[0]*img.shape[0])
        x_end= int(x_init + self.corp_ratio*img.shape[0])
        random.shuffle(portions) 
        y_init = int(portions[0]*img.shape[1])
        y_end = int(y_init + self.corp_ratio*img.shape[1])
        
        crop_img = img[x_init:x_end,y_init:y_end,:]
        
        return crop_img

    def random_flip(self, img):
        
        flip_code = [0,1,2]
        random.shuffle(flip_code) 
         
        if flip_code[0] != 2:
            flip_img = cv2.flip(img, flip_code[0])
        
        else:
            flip_img = img
        
        return flip_img



    def get_batch(self, btype, index_list,step , isflip = True):

        batch_img = []
        batch_label = []

        
        batch_size = self.conf.batch_size

        if btype == 'train':
            ann_file = self.conf.ann_train
            data_root = self.conf.training_set
            

        elif btype == 'validation':
            ann_file = self.conf.ann_val
            data_root = self.conf.validation_set
           

        elif btype == 'test':
            ann_file = self.conf.ann_test
            data_root = self.conf.test_set
            

        else:

            print("Error: Batch type shall be 'train', 'va;idation', or 'test'")
            return


        for idx in range(step*batch_size,step*batch_size + batch_size):
            
            i = index_list[idx]

            img = cv2.imread(os.path.join(data_root, 'images',ann_file['image_id'][i]))
            
            if img == None:
                print("No such file ", os.path.join(data_root, 'images',ann_file['image_id'][i]))
                return
            
            label = ann_file['label_id'][i]
            tmp_img = self.random_crop(img)
            if isflip == True: tmp_img = self.random_flip(tmp_img)   
            tmp_img = cv2.resize(tmp_img, (self.conf.img_size, self.conf.img_size))
            
            batch_img.append(tmp_img)
            batch_label.append(label)
            
        batch_img = np.stack(batch_img)
        
        return batch_img, batch_label





    
