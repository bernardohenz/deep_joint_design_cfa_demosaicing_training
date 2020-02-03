from keras.layers import Input, Conv2D, Concatenate, Add, Activation, BatchNormalization, Multiply
from keras.models import Model,Sequential
import keras
from keras import optimizers
from keras.models import model_from_json
from PIL import Image
import numpy as np
import math
import os, json
from color_filter_layers import ColorFilterLayer
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau
from custom_metrics import psnr_metric
from keras.constraints import nonneg
from color_filter_constraint import MaxMax
from sequenceGenerator_withmasks import Generator, ImageTransformations
import imageio
import argparse


def create_model_old_method_cfa_only(args):
    size_CFA  = tuple(args.size_CFA)
    number_of_colors_on_cfa = size_CFA[0] * size_CFA[1]
    number_of_filters = [ [64,64,64],[64,64,64],[128,128,128],[128,128,128]] 
    number_of_residual_blocks = len(number_of_filters)
    choosen_color_filter_constraint = MaxMax() if args.use_maxOne_constraint else nonneg()

    main_input= Input(shape=(None, None,3))
    mask_inputs= Input(shape=(None, None,number_of_colors_on_cfa))
    allInputs = [main_input]+[mask_inputs]
    if (args.use_bayer_CFA):
        assert size_CFA == (2,2),"Bayer CFA only works with 2x2 CFA pattern size."
        tmpnp = np.zeros((number_of_colors_on_cfa,3,1,1)).astype('float32')
        tmpnp[0,0,0,0] = 1
        tmpnp[1,1,0,0] = 1
        tmpnp[2,1,0,0] = 1
        tmpnp[3,2,0,0] = 1
        tmp_conv = Conv2D(number_of_colors_on_cfa, (1, 1), padding='same',\
                    weights= [tmpnp.transpose((2,3,1,0)),np.zeros(number_of_colors_on_cfa)],trainable=False,\
                    kernel_constraint=choosen_color_filter_constraint)(main_input)
    else:
        tmp_conv = Conv2D(number_of_colors_on_cfa, (1, 1), padding='same',\
            use_bias=False,kernel_constraint=choosen_color_filter_constraint)(main_input)
    submosaics = Multiply()([tmp_conv,mask_inputs])

    #Interpolating submosaics
    if (args.use_interp_layer):
        ## Interpolation Kernel
        interp_kernelA = np.expand_dims(np.append(np.arange(1,size_CFA[0]+1),np.arange(size_CFA[0]-1,0,-1)).astype('float32')/size_CFA[0],0)
        interp_kernelB = np.expand_dims(np.append(np.arange(1,size_CFA[1]+1),np.arange(size_CFA[1]-1,0,-1)).astype('float32')/size_CFA[1],0)
        interp_kernel = np.multiply(np.transpose(interp_kernelA),interp_kernelB)

        all_kernels = np.zeros((number_of_colors_on_cfa,number_of_colors_on_cfa,interp_kernel.shape[0],interp_kernel.shape[1]))
        for i in range(number_of_colors_on_cfa):
            all_kernels[i,i] = interp_kernel


        tmp_interp_output = Conv2D(number_of_colors_on_cfa,(interp_kernel.shape[0],interp_kernel.shape[1]),padding='same',
                                    weights= [all_kernels.transpose((2,3,1,0)),np.zeros(number_of_colors_on_cfa)],
                                    trainable=args.trainable_interp_layer)(submosaics)


    cfa_output_ones = np.ones((1,1,number_of_colors_on_cfa,1))
    cfa_output = Conv2D(1,(1,1),padding='same', weights= [cfa_output_ones,np.zeros(1)],trainable=False)(submosaics)
    model_encode = Model(inputs=allInputs,outputs=cfa_output)
    
    if (args.use_interp_layer):
        demosaiced_input = Concatenate(axis=-1)([cfa_output,submosaics,tmp_interp_output])
    else:
        demosaiced_input = Concatenate(axis=-1)([cfa_output,submosaics])
    current_layer = Conv2D(number_of_filters[0][0], (3, 3),padding='same',activation='relu')(demosaiced_input)
    last_res_short=current_layer
    for i in range(number_of_residual_blocks):
        current_layer = Conv2D(number_of_filters[i][0], (3, 3), padding='same')(last_res_short)
        current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        current_layer = Conv2D(number_of_filters[i][1], (3, 3), padding='same')(current_layer)
        current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        current_layer = Conv2D(number_of_filters[i][2], (3, 3), padding='same')(current_layer)
        current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        last_res_short = current_layer
    current_layer = Concatenate(axis=-1)([current_layer,demosaiced_input])
    current_layer = Conv2D(128, (3, 3), padding='same')(current_layer)
    current_layer = Activation('relu')(current_layer)
    current_layer = Conv2D(3, (3, 3), padding='same')(current_layer)

    full_model = Model(inputs=allInputs,outputs=current_layer)
    optim = optimizers.Adam()
    full_model.compile(optimizer=optim,
                loss='mse',
                metrics=[psnr_metric])
    return full_model
    


def create_generators(args):
    train_transformations = ImageTransformations(data_aug=True,final_size=(128,128))
    train_generator = Generator(data_dir=args.train_dir,size_cfa=args.size_CFA,batch_size=args.batch_size, image_transformations = train_transformations)

    val_transformations = ImageTransformations(data_aug=False,final_size=(128,128))
    val_generator = Generator(data_dir=args.val_dir,size_cfa=args.size_CFA,batch_size=args.batch_size, image_transformations = val_transformations)

    return train_generator, val_generator
   
    


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir',    help='Path to the training images.', type=str, default='../demosaic_dataset/images/train')
    parser.add_argument('--val_dir',    help='Path to the validation images.', type=str, default='../demosaic_dataset/images/val')


    parser.add_argument('--input_size',    help='Size of the input image-size.', type=int, default=128)
    parser.add_argument('--batch-size',    help='Number of patches per batch.', type=int, default=32)
    parser.add_argument('--size_CFA',      help='Size of the Color Filter Array', type=str, default='4,4')

    parser.add_argument('--use_bayer_CFA',        help='This will use weights for the Bayer CFA (size_CFA must be 2,2)', action='store_true')
    parser.add_argument('--use_maxOne_constraint',        help='Whether to use constraint in color filters to not go over 1 (default is only non-neg)', action='store_true')
    parser.add_argument('--use_interp_layer',        help='Whether to use or not interpolation layer.', type=int, default=1)
    parser.add_argument('--trainable_interp_layer',  help='Whether to make trainable the interpolation layer.', type=int, default=0)
    parser.add_argument('--concatenate_submosaics',        help='Whether to concatenate or not submosaics.', type=int, default=1)

    parser.add_argument('--number_of_residual_blocks',        help='How many residual blocks will be used (each block consists of 3 conv layers).', type=int, default=4)
    parser.add_argument('--number_of_filters',        help='Number of filters on each conv layer', type=int, default=64)
    parser.add_argument('--use_batch_norm',        help='Whether to use or not batchnormalization layer.', type=int, default=1)
    parser.add_argument('--use_shortcut_on_resblocks',        help='Whether to use or not skip connections (add) inside residual blocks.', type=int, default=1)

    parser.add_argument('--epochs',          help='Number epochs.', type=int, default=8)
    parser.add_argument('--save_path',   help='Path to store snapshots of models during training', default='./weights')


    args = parser.parse_args(args)
    args.size_CFA = [int(item) for item in args.size_CFA.split(',')]

    model = create_model_old_method_cfa_only(args)
    train_generator,val_generator = create_generators(args)

    model.fit_generator(train_generator,
                        epochs=args.epochs,
                        verbose=1,
                        validation_data=val_generator,
                        validation_steps=20,
                        callbacks=[TensorBoard(log_dir='./logs_CFA'), 
                                    ModelCheckpoint(filepath=os.path.join(args.save_path,'weights_ep-{epoch:02d}-val_{psnr_metric:.2f}.h5'), 
                                    verbose=1, 
                                    save_best_only=False, 
                                    save_weights_only=False, 
                                    monitor='psnr_metric'),
                                    ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.2, patience=3, verbose=0,
                                    )],
                        workers=12
                        )
    model.save('4x4_8epochs_nonneg.h5')
    

if __name__ == '__main__':
    main()
