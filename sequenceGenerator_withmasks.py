import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
import scipy.io
from six.moves import range
import os, glob
import sys
import threading
import copy
import inspect
import types
import keras
import cv2
import imageio
import random

from keras import backend as K
from utils.image_transforms import random_flip, random_rot90, random_shift, center_crop, random_crop


class ImageTransformations():
    def __init__(self,data_aug=False,final_size=(128,128),shift_ranges=(30,30)):
        self.final_size = final_size
        self.shift_ranges = shift_ranges
        self.data_aug = data_aug
    
    def performTransformations(self,image):
        if (self.data_aug):
            outimage = random_shift( random_flip( random_rot90(image)), self.shift_ranges)
            return outimage
        return image


def generate_default_mask(size_cfa=(4,4),patch_size=(128,128)):

    number_of_colors_on_cfa = size_cfa[0] * size_cfa[1]
    masks_train = np.zeros((1,patch_size[0],patch_size[1],number_of_colors_on_cfa)) 
    masks_train = masks_train.astype('float32')
    for i in range(0,patch_size[0],size_cfa[0]):
        for j in range(0,patch_size[1],size_cfa[1]):
            masks_train[0,i,j,0] = 1

    lastYMask = masks_train[0,:,:,0]
    mask_i_counter=0
    for i in range(0,size_cfa[0]):
        masks_train[0,:,:,mask_i_counter] = lastYMask
        mask_i_counter = mask_i_counter+1
        lastXMask = lastYMask
        for j in range(1,size_cfa[1]):
            new_mask = np.roll(lastXMask,1,axis=1)
            masks_train[0,:,:,mask_i_counter] = new_mask
            mask_i_counter = mask_i_counter+1
            lastXMask = new_mask
        new_mask = np.roll(lastYMask,1,axis=0)
        lastYMask = new_mask
    
    return masks_train

class Generator(keras.utils.Sequence):

    def __init__(self, data_dir,size_cfa=(4,4),shuffle=True,batch_size=32,image_transformations=None):
        """ Initialize a RSNA data generator.

        Args
            data_dir: Path to where the RSNA dataset is stored.
        """
        self.data_dir  = data_dir

        if os.path.exists(os.path.join(data_dir,'filelist.txt')):
            with open(os.path.join(data_dir,'filelist.txt'),'r') as f:
                self.image_names = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            self.image_names = [x.strip() for x in self.image_names] 
        else:
            self.image_names = glob.glob(os.path.join(data_dir,'**/*.png'),recursive=True)
                
        self.nb_sample = len(self.image_names)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_transformations = image_transformations

        self.masks = generate_default_mask(size_cfa=size_cfa, patch_size=image_transformations.final_size)
        self.masks = np.tile(self.masks,[batch_size,1,1,1])

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        lenn = int(np.floor(self.nb_sample) / self.batch_size)
        if (self.nb_sample % self.batch_size >0):
            lenn +=1
        return lenn
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        if (  (index+1)*self.batch_size <= len(self.indexes)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        # Find list of IDs
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_names)).astype('int32')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        batch_y = np.empty((len(list_indexes), self.image_transformations.final_size[0],self.image_transformations.final_size[1], 3))
        #if self.noise_data_dir:
        batch_x = np.empty((len(list_indexes), self.image_transformations.final_size[0],self.image_transformations.final_size[1], 3))
        # Generate data
        for i, cur_index in enumerate(list_indexes):
            # for huge datasets
            fname = os.path.join(self.data_dir,self.image_names[cur_index])
            y = self.read_image(fname)

            #for small datasets
            #x = self.loaded_imgs[cur_index]

            y = self.image_transformations.performTransformations(y)

            # Store sample
            batch_y[i] = y
            batch_x[i] = y

            ## For training with Gaussian Noise
            #choosen_var = np.random.uniform(0.0001,0.0025)
            #batch_x[i] = np.clip(random_noise(y, mode='gaussian', var= choosen_var ),0,1)

        tmp_mask = self.masks[:len(list_indexes)]
        batch_x = [batch_x,tmp_mask]
        return batch_x, batch_y

    def read_image(self,name):
        #take into account sRGB or linear          
        out = imageio.imread(name).astype('float32')/255.0
        return out
