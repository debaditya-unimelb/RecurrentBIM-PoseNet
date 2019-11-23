from keras.layers import Input, Dense, Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D, LSTM
from keras.layers import ZeroPadding2D, Dropout, Flatten, TimeDistributed
from keras.layers import merge, Reshape, Activation, BatchNormalization
from keras.utils.np_utils import convert_kernel
from keras import backend as K
#K.set_image_dim_ordering('tf')
from keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math


# define the LSTM length, beta, dropouts for the LSTM
LSTM_size = 256
beta = 600
drop1 = 0.25
drop2 = 0.25


# define the loss functions
def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lx = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    print lx
    return (0.3 * lx)

def euc_loss1q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lq = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    print lq
    return (0.3 * beta * lq)

def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lx = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    return (0.3 * lx)

def euc_loss2q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lq = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    return (0.3 * beta * lq)

def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lx = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    return (1 * lx)

def euc_loss3q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:])))
#    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
#    lq = K.sqrt(K.sum(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True), axis=0, keepdims=True)) # Sum loss
    return (beta * lq)


# creates timedistributed Posenet from GoogLeNet and adds LSTM units 
def create_posenet(weights_path=None, tune=False, stepsize=1):
    with tf.device('/gpu:0'):
        input = Input(shape=(stepsize, 224, 224, 3))
        
        conv1 = TimeDistributed(Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu'),name='conv1')(input)
        
        pool1 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='pool1')(conv1)

        norm1 = BatchNormalization(axis=-1, name='norm1')(pool1)
        
        reduction2 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='reduction2')(norm1)
        
        conv2 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu'),name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=-1, name='norm2')(conv2)
        
        pool2 = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid'),name='pool2')(norm2)

        icp1_reduction1 = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu'),name='icp1_reduction1')(pool2)

        icp1_out1 = TimeDistributed(Convolution2D(128,3,3,border_mode='same',activation='relu'),name='icp1_out1')(icp1_reduction1)
        

        icp1_reduction2 = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu'),name='icp1_reduction2')(pool2)

        icp1_out2 = TimeDistributed(Convolution2D(32,5,5,border_mode='same',activation='relu'),name='icp1_out2')(icp1_reduction2)
        

        icp1_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp1_pool')(pool2)

        icp1_out3 = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp1_out3')(icp1_pool)

       
        icp1_out0 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp1_out0')(pool2)

        
        icp2_in = merge([icp1_out0, icp1_out1, icp1_out2, icp1_out3],mode='concat',concat_axis=4,name='icp2_in')


        
        


        icp2_reduction1 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp2_reduction1')(icp2_in)

        icp2_out1 = TimeDistributed(Convolution2D(192,3,3,border_mode='same',activation='relu'),name='icp2_out1')(icp2_reduction1)
        
       
        icp2_reduction2 = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp2_reduction2')(icp2_in)

        icp2_out2 = TimeDistributed(Convolution2D(96,5,5,border_mode='same',activation='relu'),name='icp2_out2')(icp2_reduction2)


        icp2_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp2_pool')(icp2_in)

        icp2_out3 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp2_out3')(icp2_pool)


        icp2_out0 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp2_out0')(icp2_in)

        
        icp2_out = merge([icp2_out0, icp2_out1, icp2_out2, icp2_out3],mode='concat',concat_axis=4,name='icp2_out')






        icp3_in = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='icp3_in')(icp2_out)

        icp3_reduction1 = TimeDistributed(Convolution2D(96,1,1,border_mode='same',activation='relu'),name='icp3_reduction1')(icp3_in)

        icp3_out1 = TimeDistributed(Convolution2D(208,3,3,border_mode='same',activation='relu'),name='icp3_out1')(icp3_reduction1)


        icp3_reduction2 = TimeDistributed(Convolution2D(16,1,1,border_mode='same',activation='relu'),name='icp3_reduction2')(icp3_in)

        icp3_out2 = TimeDistributed(Convolution2D(48,5,5,border_mode='same',activation='relu'),name='icp3_out2')(icp3_reduction2)
        

        icp3_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp3_pool')(icp3_in)

        icp3_out3 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp3_out3')(icp3_pool)

        
        icp3_out0 = TimeDistributed(Convolution2D(192,1,1,border_mode='same',activation='relu'),name='icp3_out0')(icp3_in)
        
        
        icp3_out = merge([icp3_out0, icp3_out1, icp3_out2, icp3_out3],mode='concat',concat_axis=4,name='icp3_out')
        




        
        cls1_pool = TimeDistributed(AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid'),name='cls1_pool')(icp3_out)
        
        cls1_reduction_pose = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='cls1_reduction_pose')(cls1_pool)


        cls1_fc1_flat = TimeDistributed(Flatten())(cls1_reduction_pose)
        
#        cls1_fc1_pose = TimeDistributed(Dense(1024,activation='relu'),name='cls1_fc1_pose')(cls1_fc1_flat)

#        loss1_drop_fc = TimeDistributed(Dropout(0.7))(cls1_fc1_pose)

#        cls1_fc_pose_xyz = TimeDistributed(Dense(3),name='cls1_fc_pose_xyz')(loss1_drop_fc)
        
#        cls1_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls1_fc_pose_wpqr')(loss1_drop_fc)

        cls1_fc1_pose = TimeDistributed(Dense(1024,activation='relu'),name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = TimeDistributed(Dense(3),name='cls1_fc_pose_xyz')(cls1_fc1_pose)
        
        cls1_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls1_fc_pose_wpqr')(cls1_fc1_pose)




        
        icp4_reduction1 = TimeDistributed(Convolution2D(112,1,1,border_mode='same',activation='relu'),name='icp4_reduction1')(icp3_out)

        icp4_out1 = TimeDistributed(Convolution2D(224,3,3,border_mode='same',activation='relu'),name='icp4_out1')(icp4_reduction1)

        
        icp4_reduction2 = TimeDistributed(Convolution2D(24,1,1,border_mode='same',activation='relu'),name='icp4_reduction2')(icp3_out)

        icp4_out2 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp4_out2')(icp4_reduction2)


        icp4_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp4_pool')(icp3_out)

        icp4_out3 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp4_out3')(icp4_pool)


        icp4_out0 = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp4_out0')(icp3_out)

        
        icp4_out = merge([icp4_out0, icp4_out1, icp4_out2, icp4_out3],mode='concat',concat_axis=4,name='icp4_out')






        icp5_reduction1 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp5_reduction1')(icp4_out)

        icp5_out1 = TimeDistributed(Convolution2D(256,3,3,border_mode='same',activation='relu'),name='icp5_out1')(icp5_reduction1)


        icp5_reduction2 = TimeDistributed(Convolution2D(24,1,1,border_mode='same',activation='relu'),name='icp5_reduction2')(icp4_out)

        icp5_out2 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp5_out2')(icp5_reduction2)


        icp5_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp5_pool')(icp4_out)

        icp5_out3 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp5_out3')(icp5_pool)


        icp5_out0 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp5_out0')(icp4_out)

        
        icp5_out = merge([icp5_out0, icp5_out1, icp5_out2, icp5_out3],mode='concat',concat_axis=4,name='icp5_out')






        icp6_reduction1 = TimeDistributed(Convolution2D(144,1,1,border_mode='same',activation='relu'),name='icp6_reduction1')(icp5_out)

        icp6_out1 = TimeDistributed(Convolution2D(288,3,3,border_mode='same',activation='relu'),name='icp6_out1')(icp6_reduction1)

        
        icp6_reduction2 = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp6_reduction2')(icp5_out)

        icp6_out2 = TimeDistributed(Convolution2D(64,5,5,border_mode='same',activation='relu'),name='icp6_out2')(icp6_reduction2)

        
        icp6_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp6_pool')(icp5_out)

        icp6_out3 = TimeDistributed(Convolution2D(64,1,1,border_mode='same',activation='relu'),name='icp6_out3')(icp6_pool)


        icp6_out0 = TimeDistributed(Convolution2D(112,1,1,border_mode='same',activation='relu'),name='icp6_out0')(icp5_out)

        
        icp6_out = merge([icp6_out0, icp6_out1, icp6_out2, icp6_out3],mode='concat',concat_axis=4,name='icp6_out')
       



        cls2_pool = TimeDistributed(AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid'),name='cls2_pool')(icp6_out)

        cls2_reduction_pose = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='cls2_reduction_pose')(cls2_pool)


        cls2_fc1_flat = TimeDistributed(Flatten())(cls2_reduction_pose)

#        cls2_fc1 = TimeDistributed(Dense(1024,activation='relu'),name='cls2_fc1')(cls2_fc1_flat)

#        loss2_drop_fc = TimeDistributed(Dropout(0.7))(cls2_fc1)
        
#        cls2_fc_pose_xyz = TimeDistributed(Dense(3),name='cls2_fc_pose_xyz')(loss2_drop_fc)
        
#        cls2_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls2_fc_pose_wpqr')(loss2_drop_fc)    

        cls2_fc1 = TimeDistributed(Dense(1024,activation='relu'),name='cls2_fc1')(cls2_fc1_flat)
        
        cls2_fc_pose_xyz = TimeDistributed(Dense(3),name='cls2_fc_pose_xyz')(cls2_fc1)
        
        cls2_fc_pose_wpqr = TimeDistributed(Dense(4),name='cls2_fc_pose_wpqr')(cls2_fc1)   





        icp7_reduction1 = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp7_reduction1')(icp6_out)

        icp7_out1 = TimeDistributed(Convolution2D(320,3,3,border_mode='same',activation='relu'),name='icp7_out1')(icp7_reduction1)


        icp7_reduction2 = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp7_reduction2')(icp6_out)

        icp7_out2 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp7_out2')(icp7_reduction2)


        icp7_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp7_pool')(icp6_out)

        icp7_out3 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp7_out3')(icp7_pool)

        
        icp7_out0 = TimeDistributed(Convolution2D(256,1,1,border_mode='same',activation='relu'),name='icp7_out0')(icp6_out)
        

        icp7_out = merge([icp7_out0, icp7_out1, icp7_out2, icp7_out3],mode='concat',concat_axis=4,name='icp7_out')

        
        


        
        icp8_in = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same'),name='icp8_in')(icp7_out)
        
        icp8_reduction1 = TimeDistributed(Convolution2D(160,1,1,border_mode='same',activation='relu'),name='icp8_reduction1')(icp8_in)

        icp8_out1 = TimeDistributed(Convolution2D(320,3,3,border_mode='same',activation='relu'),name='icp8_out1')(icp8_reduction1)


        icp8_reduction2 = TimeDistributed(Convolution2D(32,1,1,border_mode='same',activation='relu'),name='icp8_reduction2')(icp8_in)

        icp8_out2 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp8_out2')(icp8_reduction2)


        icp8_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp8_pool')(icp8_in)

        icp8_out3 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp8_out3')(icp8_pool)

        
        icp8_out0 = TimeDistributed(Convolution2D(256,1,1,border_mode='same',activation='relu'),name='icp8_out0')(icp8_in)
        
        icp8_out = merge([icp8_out0, icp8_out1, icp8_out2, icp8_out3],mode='concat',concat_axis=4,name='icp8_out')
        





        icp9_reduction1 = TimeDistributed(Convolution2D(192,1,1,border_mode='same',activation='relu'),name='icp9_reduction1')(icp8_out)

        icp9_out1 = TimeDistributed(Convolution2D(384,3,3,border_mode='same',activation='relu'),name='icp9_out1')(icp9_reduction1)


        icp9_reduction2 = TimeDistributed(Convolution2D(48,1,1,border_mode='same',activation='relu'),name='icp9_reduction2')(icp8_out)

        icp9_out2 = TimeDistributed(Convolution2D(128,5,5,border_mode='same',activation='relu'),name='icp9_out2')(icp9_reduction2)


        icp9_pool = TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same'),name='icp9_pool')(icp8_out)

        icp9_out3 = TimeDistributed(Convolution2D(128,1,1,border_mode='same',activation='relu'),name='icp9_out3')(icp9_pool)

        
        icp9_out0 = TimeDistributed(Convolution2D(384,1,1,border_mode='same',activation='relu'),name='icp9_out0')(icp8_out)
        
        icp9_out = merge([icp9_out0, icp9_out1, icp9_out2, icp9_out3],mode='concat',concat_axis=4,name='icp9_out')
        
        




        cls3_pool = TimeDistributed(AveragePooling2D(pool_size=(7,7),strides=(1,1),border_mode='valid'),name='cls3_pool')(icp9_out)

        cls3_fc1_flat = TimeDistributed(Flatten())(cls3_pool)

#        cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu'),name='cls3_fc1_pose')(cls3_fc1_flat)

#        loss3_drop_fc = TimeDistributed(Dropout(0.5))(cls3_fc1_pose)

#        lstm = (LSTM(LSTM_size ,return_sequences=True, input_shape=(stepsize, 2048)))(loss3_drop_fc)
        
#        cls3_fc_pose_xyz_new = TimeDistributed(Dense(3),name='cls3_fc_pose_xyz_new')(lstm)
        
#        cls3_fc_pose_wpqr_new = TimeDistributed(Dense(4),name='cls3_fc_pose_wpqr_new')(lstm)

        cls3_fc1_pose = TimeDistributed(Dense(2048,activation='relu'),name='cls3_fc1_pose')(cls3_fc1_flat)

        lstm = LSTM(LSTM_size, dropout_W=drop1, dropout_U=drop2, return_sequences=True, input_shape=(stepsize, 2048))(cls3_fc1_pose)
        
        cls3_fc_pose_xyz_new = TimeDistributed(Dense(3),name='cls3_fc_pose_xyz_new')(lstm)
        
        cls3_fc_pose_wpqr_new = TimeDistributed(Dense(4),name='cls3_fc_pose_wpqr_new')(lstm)
        

        


        
        posenet = Model(input=input, output=[cls1_fc_pose_xyz, cls1_fc_pose_wpqr, cls2_fc_pose_xyz, cls2_fc_pose_wpqr, cls3_fc_pose_xyz_new, cls3_fc_pose_wpqr_new])
    
    if tune:
        if weights_path:
            posenet.load_weights(weights_path,by_name=True)
    
    return posenet


if __name__ == "__main__":
	print("Run train.py to train and test.py to test the network")