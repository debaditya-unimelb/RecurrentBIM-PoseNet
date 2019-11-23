from tqdm import tqdm
import numpy as np
import os.path
import sys
import random
import math
import cv2

# define the path to the dataset and the files containing the train and test ground truths
directory = '/path/to/the/dataset/Gradmag-Syn-Car/'
dataset_train = 'groundtruth_GradmagSynCar.txt'
dataset_test = 'groundtruth_GradmagReal.txt'

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

# function for performing centre croppping
def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	cropped_img = img[height_offset:height_offset + output_side_length,
	                          width_offset:width_offset + output_side_length]
	return cropped_img

	
# function for croppping and mean subtraction train images
def preprocess_train(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		X = cv2.resize(X, (320, 240))
		X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out.append(Y)
	return images_out


# function for croppping and mean subtraction test images
def preprocess_test(images_train, images_test):
	images_out_test = [] #final result
	#Resize and crop and compute mean!
	images_cropped_train = []
	images_cropped_test = []

	for i in tqdm(range(len(images_train))):
		X = cv2.imread(images_train[i])
		X = cv2.resize(X, (320, 240))
		X = centeredCrop(X, 224)
		images_cropped_train.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped_train):
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N

	for i in tqdm(range(len(images_test))):
		X = cv2.imread(images_test[i])
		X = cv2.resize(X, (320, 240))
		X = centeredCrop(X, 224)
		images_cropped_test.append(X)
	#Subtract mean from all images
	for X in tqdm(images_cropped_test):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out_test.append(Y)

	return images_out_test


# reading ground truth data from files
def get_data(dataset_train, dataset_test):
	poses_train = []
	images_train = []
	poses_test = []
	images_test = []

	with open(directory+dataset_train) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses_train.append((p0,p1,p2,p3,p4,p5,p6))
			images_train.append(directory+fname)
	images_out_train = preprocess_train(images_train)

	with open(directory+dataset_test) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses_test.append((p0,p1,p2,p3,p4,p5,p6))
			images_test.append(directory+fname)
	images_out_test = preprocess_test(images_train, images_test)

	return datasource(images_out_train, poses_train), datasource(images_out_test, poses_test)

# creating the final datasource containing images and the respective camera poses for training and testing datasets
def getKings():
	datasource_train, datasource_test = get_data(dataset_train, dataset_test)

	images_train = []
	poses_train = []

	images_test = []
	poses_test = []


	for i in range(len(datasource_train.images)):
		# print(i)
		images_train.append(datasource_train.images[i])
		poses_train.append(datasource_train.poses[i])

	for i in range(len(datasource_test.images)):
		# print(i)
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_train, poses_train), datasource(images_test, poses_test)