
import keras.backend as K
import tensorflow as tf

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true,y_pred,1)