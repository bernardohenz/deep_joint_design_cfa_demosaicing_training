import keras
from keras.models import Model,model_from_json
from keras.constraints import nonneg
from PIL import Image
import numpy as np
import glob, os, time, random, math, sys
import scipy
import argparse, json
from argparse import Namespace
from keras import backend as K
from custom_metrics import psnr_metric

from utils.utils_test import predictImg,cpsnr,reject_outliers,predictHugeImg,predictHugeImg

autoencoder = keras.models.load_model('4x4_8epochs_nonneg.h5',custom_objects={'psnr_metric': psnr_metric})
print('Loaded model!')

pattern_CFA = [4,4]

datasets = ['kodak', 'mcm', 'hdrvdp', 'moire']

noise_std = 0

print('Starting predictions on datasets: ',str(datasets))

for cur_dataset in datasets:
   imgs = glob.glob('datasets/'+cur_dataset+'/*')
   psnrs = np.zeros((len(imgs)))
   times = np.zeros((len(imgs)))
   for i,img_name in enumerate(imgs):
      img = np.asarray(Image.open(img_name)).astype('float32')
      if (noise_std >0):
         predicted,times[i] = predictImgNoise(img,autoencoder,opt.noise_std)
      else:
         predicted,times[i] = predictImg(img,autoencoder)
      psnrs[i] = cpsnr(img,predicted)
   print("{:s} - psnr: {:.2f} time : {:.2f} seg".format(cur_dataset,np.mean(psnrs),np.mean(reject_outliers(times))))
