import keras

import helper_train_mean as helper
#import helper

#import posenet_dropout as posenet
#import posenet_dropout_regu as posenet
import posenetLSTMDropout as posenet
#import posenetLSTMnoSequence as posenet
#import posenetLSTM as posenet

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random


# define the setting of the training. Note some other setting of the network can be found in the begenning of the posenetLSTMDropout.py file
window=10
learningRate = 0.001
batch_size = 25
beta=posenet.beta
LSTM_size=posenet.LSTM_size
drop1 = posenet.drop1
drop2 = posenet.drop2

#def rmse(y_true, y_pred):
#    from keras import backend
#    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

# function to check the loss of the last branch of the network
class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y, z = self.test_data
        one, two, three, four, five, six = self.model.predict(x, verbose=0)
        diff_sum = np.sum(np.absolute(np.squeeze(y) - np.squeeze(five)))
        print diff_sum/25
        return diff_sum/25
#        print('\nTesting loss: {}, acc: {}\n'.format(difference, difference))

# function to retrn the loss
def validation_error_x(y_true, y_pred):
    loss_x = keras.backend.sum(keras.backend.abs(y_true - y_pred))
    return loss_x


if __name__ == "__main__":

    # Create model and load the weights of GoogLeNet (Trained on Places)
    model = posenet.create_posenet('googlenet_weights.h5', True, window)

#    for layer in model.layers[:-15]:
#        layer.trainable = False
#        if isinstance(layer, keras.layers.normalization.BatchNormalization):
#            layer._per_input_updates = {}
#            print ("BATCH NORM FROZEN")
#        print layer.name
#    print(model.summary())

	#define optimiser, compile model and get the training dataset	
    adam = Adam(lr=learningRate, clipvalue=1.5)
	
	#compile the model with the custom loss function
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
#                                        'cls3_fc_pose_xyz_new': posenet.euc_loss3x, 'cls3_fc_pose_wpqr_new': posenet.euc_loss3q})
                                        'cls3_fc_pose_xyz_new': posenet.euc_loss3x, 'cls3_fc_pose_wpqr_new': posenet.euc_loss3q}, metrics=[validation_error_x])

	# get the training and testing data (images with ground truth)
    dataset_train, dataset_test = helper.getKings()

	# Process the array 
    X_train = np.squeeze(np.array(dataset_train.images))
    y_train = np.squeeze(np.array(dataset_train.poses))
    X_train_new_new = np.expand_dims(X_train, axis=1)
    y_train_new_new = np.expand_dims(y_train, axis=1)

	# identify the number of samples to generate
    sequence_length_train = (len(X_train) - window + 1)
    indices = range(sequence_length_train)
	
	# shuffle training sequences
    random.shuffle(indices) # To turn off for LSTM stateful operation

	# preallocate the arrays
    X_train_shuffle = np.zeros ((sequence_length_train, window, 224,224,3), dtype=np.float64)
    y_train_shuffle = np.zeros ((sequence_length_train, window, 7), dtype=np.float64)

	# generate the image sequences and their respective ground truths
    j = 0
    for i in indices:
        for k in range (window):
            X_train_shuffle [j,k,:,:,:] = X_train_new_new[i+k,0,:,:,:]
            y_train_shuffle [j,k,:] = y_train_new_new [i+k,0,:]
        j=j+1
    print X_train_shuffle.shape, y_train_shuffle.shape
	
	# allocate arrays to separate XYZ and wpqr
    y_train_x = y_train_shuffle[:,:,0:3]
    y_train_q = y_train_shuffle[:,:,3:7]

	# Repeat all steps for test images
    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))
    X_test_new_new = np.expand_dims(X_test, axis=1)
    y_test_new_new = np.expand_dims(y_test, axis=1)

    sequence_length_test = (len(X_test) - window + 1)

    indices = range(sequence_length_test)
    random.shuffle(indices) # To turn off for LSTM stateful operation
    X_test_shuffle = np.zeros ((sequence_length_test, window, 224,224,3), dtype=np.float64)
    y_test_shuffle = np.zeros ((sequence_length_test, window, 7), dtype=np.float64)

    j = 0
    for i in indices:
        for k in range (window):
            X_test_shuffle [j,k,:,:,:] = X_test_new_new [i+k,0,:,:,:]
            y_test_shuffle [j,k,:]= y_test_new_new [i+k,0,:]
        j=j+1
	
    y_test_x = y_test_shuffle[:,:,0:3]
    y_test_q = y_test_shuffle[:,:,3:7]

    # Setup checkpointing for keeping the best results
#    checkpointer = ModelCheckpoint(filepath="today_batch25_LR0001_beta_600_brforgradmag_dropout.h5", verbose=1, save_best_only=True, save_weights_only=True)
    checkpointer = ModelCheckpoint(filepath='window'+str(window)+'batch'+str(batch_size)+'LR'+str(learningRate)+'beta'+str(beta)+'LSTM'+str(LSTM_size)+'Dropout'+str(drop1)+str(drop2)+'.h5', verbose=1, save_best_only=True, save_weights_only=True, monitor='val_cls3_fc_pose_xyz_new_validation_error_x', mode='min')

	# creating history object to train and record the log of the process
    history = model.fit(X_train_shuffle, [y_train_x, y_train_q, y_train_x, y_train_q, y_train_x, y_train_q],
          batch_size=batch_size,
          nb_epoch=400,
          validation_data=(X_test_shuffle, [y_test_x, y_test_q, y_test_x, y_test_q, y_test_x, y_test_q]),
          callbacks=[checkpointer, TestCallback([X_test_shuffle, y_test_x, y_test_q])])
		  
#    history_dict = history.history
#    print history_dict.keys()
#    print history.history['val_cls3_fc_pose_xyz_new_acc']

	#Store the loss with each iteration
    with open('window'+str(window)+'batch'+str(batch_size)+'LR'+str(learningRate)+'beta'+str(beta)+'LSTM'+str(LSTM_size)+'Dropout'+str(drop1)+str(drop2)+'.csv',"a+") as f:
          for ii in range(len(history.history['loss'])):
              f.write('{},{}\n'.format(str(history.history['loss'][ii]), str(history.history['val_loss'][ii])))

	# save the final model
    model.save_weights('window'+str(window)+'batch'+str(batch_size)+'LR'+str(learningRate)+'beta'+str(beta)+'LSTM'+str(LSTM_size)+'Dropout'+str(drop1)+str(drop2)+'_weight.h5')