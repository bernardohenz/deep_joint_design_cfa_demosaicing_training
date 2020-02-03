from keras import backend as K
from keras.engine.topology import Layer
import keras
import numpy as np
import tensorflow as tf
from keras.utils import conv_utils

class ColorFilterLayer(Layer):

    def __init__(self, cfa_shape, kernel_constraint=None,**kwargs):
        self.cfa_shape = cfa_shape
        self.num_of_filters = cfa_shape[0]*cfa_shape[1]
        self.x_step = cfa_shape[0]
        self.y_step = cfa_shape[1]
        self.kernel_size = conv_utils.normalize_tuple((1,1), 2, 'kernel_size')
        self.kernel_constraint = kernel_constraint
        super(ColorFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.num_of_filters)
        tmpmasks = np.zeros((self.x_step*self.y_step,)+self.cfa_shape)
        tmpmasks[0,0,0] = 1

        lastYMask = tmpmasks[0]
        mask_i_counter=0
        for i in range(0,self.x_step):
            tmpmasks[mask_i_counter] = lastYMask
            mask_i_counter = mask_i_counter+1
            lastXMask = lastYMask
            for j in range(1,self.y_step):
                new_mask = np.roll(lastXMask,1,axis=1)
                tmpmasks[mask_i_counter] = new_mask
                mask_i_counter = mask_i_counter+1
                lastXMask = new_mask
            new_mask = np.roll(lastYMask,1,axis=0)
            lastYMask = new_mask

        self.masks = tf.convert_to_tensor(tmpmasks,np.float32)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=kernel_shape,
                                      initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        super(ColorFilterLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        out = K.conv2d(x,self.kernel)
        shape_x = tf.shape(out)
        shape_masks = tf.shape(self.masks)
        #tiling masks to patch_size
        reshape_size = (1+shape_x[1]/shape_masks[1],1+shape_x[2]/shape_masks[2])
        reshape_size = tf.concat([[1],tf.to_int32(reshape_size)],axis=0)
        reshaped_masks = tf.to_float(K.tile(self.masks,reshape_size)[:,:shape_x[1],:shape_x[2]])
        reshaped_masks = tf.expand_dims(K.permute_dimensions(reshaped_masks,(1,2,0)),0)
        return out*reshaped_masks


    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.num_of_filters)

    def get_config(self):
        config = {
            'cfa_shape': self.cfa_shape,
            'kernel_constraint':self.kernel_constraint
        }
        return dict(list(config.items()))