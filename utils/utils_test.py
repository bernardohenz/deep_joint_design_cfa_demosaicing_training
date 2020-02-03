import numpy as np
import time
import math
from keras.layers import Input, Conv2D, Concatenate, Add, Activation, BatchNormalization
from keras.models import Model,Sequential
from color_filter_constraint import MaxMax
from color_filter_layers import ColorFilterLayer
from tqdm import tqdm

def mse(predictions,targets):
    return np.sum(((predictions - targets) ** 2))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2])

def cpsnr(img1, img2):
    mse_tmp = mse(np.round(np.clip(img1,0,255)),np.round(np.clip(img2,0,255)))
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse_tmp)

def generateMaskForImg(img,pattern_CFA):
    fixing_size = pattern_CFA-np.array([img.shape[0]%pattern_CFA[0],img.shape[1]%pattern_CFA[1]])
    fixing_size[0]= fixing_size[0]%pattern_CFA[0]
    fixing_size[1]= fixing_size[1]%pattern_CFA[1]
    mask = np.zeros((1,pattern_CFA[0]*pattern_CFA[1],img.shape[0]+fixing_size[0],img.shape[1]+fixing_size[1]))
    mask = mask.astype('float32')
    for i in range(0,img.shape[0],pattern_CFA[0]):
        for j in range(0,img.shape[1],pattern_CFA[1]):
            mask[0,0,i,j] = 1

    lastYMask = mask[0,0]
    mask_i_counter=0
    for i in range(0,pattern_CFA[0]):
        mask[0,mask_i_counter] = lastYMask
        mask_i_counter = mask_i_counter+1
        lastXMask = lastYMask
        for j in range(1,pattern_CFA[1]):
            new_mask = np.roll(lastXMask,1,axis=1)
            mask[0,mask_i_counter] = new_mask
            mask_i_counter = mask_i_counter+1
            lastXMask = new_mask
        new_mask = np.roll(lastYMask,1,axis=0)
        lastYMask = new_mask

    mask = mask[:,:,:img.shape[0],:img.shape[1]]
    return mask.transpose(0,2,3,1)

def predictImg(img,autoencoder,pattern_CFA=[4,4]):
    img2 = img.astype('float32')[:,:,:3]/255.0
    mask = generateMaskForImg(img,pattern_CFA)
    img2 = np.expand_dims(img2,0)
    start = time.time()
    prediction = autoencoder.predict([img2,mask])
    end = time.time()
    out = np.round(np.clip(prediction[0]*255,0,255))
    return out,end-start

def predictHugeImg(img,autoencoder,tile_size=1024, pattern_CFA=[4,4]):
    h,w = img.shape[:2]
    psize=tile_size
    psize = min(min(psize,h),w)
    patch_step = psize
    mask = generateMaskForImg(np.zeros((tile_size,tile_size,3)),pattern_CFA)
    img2 = img.astype('float32')[:,:,:3]/255.0
    
    R = np.zeros(img2.shape, dtype = np.float32)
    rangex = range(0,w,patch_step)
    rangey = range(0,h,patch_step)
    ntiles = len(rangex)*len(rangey)

    for start_x in rangex:
        for start_y in rangey:
            end_x = start_x+psize
            end_y = start_y+psize
            if end_x > w:
                end_x = w
                start_x = end_x-psize
            if end_y > h:
                end_y = h
                start_y = end_y-psize
                
            tileM = img2[start_y:end_y, start_x:end_x, :] 
            tileM = tileM[np.newaxis,:,:,:]
            prediction = autoencoder.predict([tileM,mask[:,:tileM.shape[1],:tileM.shape[2],:]])[0]
            s1 = prediction.shape[0]
            s2 = prediction.shape[1]
            R[start_y:start_y+s1,start_x:start_x+s2,:] = prediction
    return np.clip(np.round(R*255.0),0,255)


def predictImgNoise(img,autoencoder,noise_std=4):
    img2 = img.astype('float32')[:,:,:3]/255.0
    mask = generateMaskForImg(img,pattern_CFA)
    img2 = np.expand_dims(img2,0)
    img2 = np.clip(img2+ np.random.normal(0,noise_std/255.0,img2.shape),0,1)
    start = time.time()
    prediction = autoencoder.predict([img2,mask])
    end = time.time()
    out = np.round(np.clip(prediction[0]*255,0,255))
    return out,end-start


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
