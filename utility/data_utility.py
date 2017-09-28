import pandas as pd
import numpy as np
import cv2
import os
import random
import config
from keras.utils import np_utils


class data_utility: 
    def __init__(self, congfiguration, corp_ratio = 0.8, corp_step = 0.2):

        self.conf = config.config(congfiguration)
        self.corp_ratio = corp_ratio
        self.corp_step = corp_step

    def solid_crop(self,img):

        portions = []
        init_point = 0
        
        
        while init_point < 1- self.corp_ratio + 0.1*self.corp_step:
            
            portions.append(init_point)
            
            init_point = init_point + self.corp_step
        
        
        crop_img = []

    
        for i in range(len(portions)):
            for j in range(len(portions)):
    
                x_init = int(portions[i]*img.shape[0])
                x_end= int(x_init + self.corp_ratio*img.shape[0])
                y_init = int(portions[j]*img.shape[1])
                y_end = int(y_init + self.corp_ratio*img.shape[1])

                tmp_img = img[x_init:x_end,y_init:y_end,:]
                tmp_img = cv2.resize(tmp_img, (self.conf.img_size, self.conf.img_size))
                crop_img.append(tmp_img)
                
        tmp_img = self.center_crop(img)
        tmp_img = cv2.resize(tmp_img, (self.conf.img_size, self.conf.img_size))
        crop_img.append(tmp_img)
        crop_img = np.stack(crop_img)
        return crop_img



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

    def center_crop(self,img):

        portions = []
        init_point = 0
        
        
        x_init = int(img.shape[0]*0.1)
        x_end= int(img.shape[0]*0.9)
        y_init = int(img.shape[1]*0.1)
        y_end = int(img.shape[1]*0.9)
        
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



    def get_batch(self, btype, index_list,step , isflip = True, randpm = True):

        batch_img = []
        batch_label = []

        
        

        if btype == 'train':
            batch_size = self.conf.batch_size
            ann_file = self.conf.ann_train
            data_root = self.conf.training_set
            

        elif btype == 'validation':
            batch_size = self.conf.test_batch_size
            ann_file = self.conf.ann_val
            data_root = self.conf.validation_set
           

        elif btype == 'test':
            batch_size = self.conf.test_batch_size
            ann_file = self.conf.ann_test
            data_root = self.conf.test_set
            

        else:

            print("Error: Batch type shall be 'train', 'validation', or 'test'")
            return

        if step == 0 and random == True:
             random.shuffle(index_list) 
             print("Random shuffle")

        for idx in range(step*batch_size,step*batch_size + batch_size):
            
            i = index_list[idx]

            img = cv2.imread(os.path.join(data_root, 'images',ann_file['image_id'][i]))
            
            if img == None:
                print("No such file ", os.path.join(data_root, 'images',ann_file['image_id'][i]))
                return
            

            label = ann_file['label_id'][i]

            if btype == 'train':

                tmp_img = self.random_crop(img)
                if isflip == True: tmp_img = self.random_flip(tmp_img)   
            else:
                tmp_img = img

            tmp_img = cv2.resize(tmp_img, (self.conf.img_size, self.conf.img_size))
           

            batch_img.append(tmp_img)
            batch_label.append(label)
            
        batch_img = np.stack(batch_img)
        
        return batch_img, batch_label


    def get_batch_onehot(self, btype, index_list,step , isflip = True):

            batch_img = []
            batch_label = []

            
            if btype == 'train':
                batch_size = self.conf.batch_size
                ann_file = self.conf.ann_train
                data_root = self.conf.training_set
                

            elif btype == 'validation':
                batch_size = self.conf.test_batch_size
                ann_file = self.conf.ann_val
                data_root = self.conf.validation_set
               

            elif btype == 'test':
                batch_size = self.conf.test_batch_size
                ann_file = self.conf.ann_test
                data_root = self.conf.test_set
                

            else:

                print("Error: Batch type shall be 'train', 'validation', or 'test'")
                return


            for idx in range(step*batch_size,step*batch_size + batch_size):
                
                i = index_list[idx]

                img = cv2.imread(os.path.join(data_root, 'images',ann_file['image_id'][i]))
                
                if img is None:
                    print("No such file ", os.path.join(data_root, 'images',ann_file['image_id'][i]))
                    return
                

                label = ann_file['label_id'][i]

                if btype == 'train':
                    tmp_img = self.random_crop(img)
                    if isflip == True: tmp_img = self.random_flip(tmp_img)   
                else:
                    tmp_img = img

                tmp_img = cv2.resize(tmp_img, (self.conf.img_size, self.conf.img_size))
               

                batch_img.append(tmp_img)
                batch_label.append(np_utils.to_categorical(label, 80))
                
            batch_img = np.stack(batch_img)
            batch_label = np.vstack(batch_label)

            print(batch_label)
            
            return batch_img, batch_label



    def get_ensemble_batch(self, index_list,step , isflip = True):

        batch_size = self.conf.batch_size
        ann_file = self.conf.ann_val
        data_root = self.conf.validation_set

        batch_label = []
        
        i = index_list[step]

        img = cv2.imread(os.path.join(data_root, 'images',ann_file['image_id'][i]))
            
        if img == None:
            print("No such file ", os.path.join(data_root, 'images',ann_file['image_id'][i]))
            return


        label = ann_file['label_id'][i]
        batch_label.append(label)
        batch_img = self.solid_crop(img)

        return batch_img, batch_label
