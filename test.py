import keras
print (keras.__version__)

import math

#import helper
import helper_train_mean as helper

#import posenet
import posenetLSTMDropout as posenet
#import posenet_dropout as posenet
#import posenet_dropout_regu as posenet

import numpy as np
from keras.optimizers import Adam


# Define window length
window =10

if __name__ == "__main__":
    # load model
    model = posenet.create_posenet(None, True, window)
	
	# load weight file
    weight_file = 'window10batch25LR0.001beta600LSTM256Dropout0.50.5'
    model.load_weights(weight_file+'.h5')
	
	# define the optimiser
    adam = Adam(lr=0.001, clipvalue=1.5)
	
	#compile the model with the custom loss function
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz_new': posenet.euc_loss3x, 'cls3_fc_pose_wpqr_new': posenet.euc_loss3q})
	
	# get the training and testing data (images with ground truth)
    dataset_train, dataset_test = helper.getKings()
	
	# Process the array 
    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))
    X_test_new_new = np.expand_dims(X_test, axis=1)
    y_test_new_new = np.expand_dims(y_test, axis=1)

	# identify the number of samples to generate
    sequence_length_test = (len(X_test) - window + 1)
    indices = range(sequence_length_test)
#    random.shuffle(indices)

	# preallocate the arrays
    X_test_shuffle = np.zeros ((sequence_length_test, window, 224,224,3), dtype=np.float64)
    y_test_shuffle = np.zeros ((sequence_length_test, window, 7), dtype=np.float64)
	
	# generate the image sequences and their respective ground truths
    j = 0
    for i in indices:
        for k in range (window):
            X_test_shuffle [j,k,:,:,:] = X_test_new_new [i+k,0,:,:,:]
            y_test_shuffle [j,k,:]= y_test_new_new [i+k,0,:]
        j=j+1
	
	# Estimate the camera poses using the model loaded
    testPredict = model.predict(X_test_shuffle)

	# take the predictions from the final branch of the network
    valsx = testPredict[4]
    valsq = testPredict[5]


    # compute resultss
	
	# preallocate the arrays
    results = np.zeros((sequence_length_test,2))
    results_loc = np.zeros((sequence_length_test, 6 * window))
    results_rot = np.zeros((sequence_length_test, 8 * window))

	# check the errors by comparing with the ground truth
    for i in range(sequence_length_test):

        pose_q= np.asarray(y_test_shuffle[i, :, 3:7])
        pose_x= np.asarray(y_test_shuffle[i, :, 0:3])
        predicted_x = valsx[i]
        predicted_q = valsq[i]
        pose_q = np.squeeze(pose_q)
        pose_x = np.squeeze(pose_x)
        predicted_q = np.squeeze(predicted_q)
        predicted_x = np.squeeze(predicted_x)
        print predicted_x.shape, pose_x.shape

        # compute Individual Sample Errors
        q1 = np.zeros((window, 4))
        q2 = np.zeros((window, 4))
        d = np.zeros((window, 1))
        theta = np.zeros((window, 1))
        error_x = np.zeros((window, 1))

		# compute errors for a window length of 1
        if window == 1:
            for j in range(window):
                q1 = pose_q / np.linalg.norm(pose_q)
                q2 = predicted_q / np.linalg.norm(predicted_q)
                d = abs(np.sum(np.multiply(q1,q2)))
                theta = 2 * np.arccos(d) * 180/math.pi
                error_x = np.linalg.norm(pose_x-predicted_x)
        else:
            for j in range(window):
                q1[j,:] = pose_q[j,:] / np.linalg.norm(pose_q[j,:])
                q2[j,:] = predicted_q[j,:] / np.linalg.norm(predicted_q[j,:])
                d[j] = abs(np.sum(np.multiply(q1[j,:],q2[j,:])))
                theta[j] = 2 * np.arccos(d[j]) * 180/math.pi
                error_x[j] = np.linalg.norm(pose_x[j,:]-predicted_x[j,:])

		# print and store the results
        print pose_x, predicted_x
        error_x = np.median(error_x)
        theta = np.median(theta)
        results[i,:] = [error_x,theta]
		
		# store the results in arrays in an ordered way for saving in a text file
        for j in range (window):
            if window == 1:
                results_loc [i,3*j:(3*j+3)] = predicted_x
                results_rot[i,4*j:(4*j+4)] = predicted_q
                results_loc [i,(3*j+3*window):(3*j+3+3*window)] = pose_x
                results_rot[i,(4*j+4*window):(4*j+4+4*window)] = pose_q
            else:
                results_loc [i,3*j:(3*j+3)] = predicted_x[j,:]
                results_rot[i,4*j:(4*j+4)] = predicted_q[j,:]
                results_loc [i,(3*j+3*window):(3*j+3+3*window)] = pose_x[j,:]
                results_rot[i,(4*j+4*window):(4*j+4+4*window)] = pose_q[j,:]
		
		# print the final errors and the iteration of sequence
        print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta
    median_result = np.median(results,axis=0)
	
	# print the median errors
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')
	
	# save the location and rotations of the sequences in an ordered manner
    np.savetxt(weight_file+'Window'+str(window)+'location.txt', results_loc, delimiter=' ')
    np.savetxt(weight_file+'Window'+str(window)+'rotation.txt', results_rot, delimiter=' ')