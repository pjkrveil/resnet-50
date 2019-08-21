import os
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib as plt
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
	"""
	Implementation of the identity block.

	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	f -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network

	Returns:
	X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
	"""

	# defining name basis
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	# Retrieve Filters
	F1, F2, F3 = filters

	# Save the input value. You'll need this later to add back to the main path. 
	X_shortcut = X

	# First component of main path
	X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path
	X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path
	X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)


	return X


def convolutional_block(X, f, filters, stage, block, s = 2):
	"""
	Implementation of the convolutional block.

	Arguments:
	X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	f -- integer, specifying the shape of the middle CONV's window for the main path
	filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	stage -- integer, used to name the layers, depending on their position in the network
	block -- string/character, used to name the layers, depending on their position in the network
	s -- Integer, specifying the stride to be used

	Returns:
	X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
	"""

	# defining name basis
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	# Retrieve Filters
	F1, F2, F3 = filters

	# Save the input value
	X_shortcut = X


	##### MAIN PATH #####
	# First component of main path 
	X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path
	X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path
	X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

	##### SHORTCUT PATH ####
	X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
	X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)

	return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
	"""
	Implementation of the popular ResNet50 the following architecture:
	CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
	-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

	Arguments:
	input_shape -- shape of the images of the dataset
	classes -- integer, number of classes

	Returns:
	model -- a Model() instance in Keras
	"""

	# Define the input as a tensor with shape input_shape
	X_input = Input(input_shape)


	# Zero-Padding
	X = ZeroPadding2D((3, 3))(X_input)

	# Stage 1
	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Stage 2
	X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
	X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
	X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

	# Stage 3
	X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
	X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

	# Stage 4
	X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
	X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

	# Stage 5
	X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
	X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
	X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

	# AVGPOOL
	X = AveragePooling2D(pool_size = (2, 2), strides=None, padding='valid', name='avg_pool')(X)

	# output layer
	X = Flatten()(X)
	X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


	# Create model
	model = Model(inputs = X_input, outputs = X, name='ResNet50')

	return model



def main():

	"""
	# Test identity_block part
	tf.compat.v1.get_default_graph()
	tf.reset_default_graph()

	with tf.Session() as test:
		np.random.seed(1)
		A_prev = tf.placeholder("float", [3, 4, 4, 6])
		X = np.random.randn(3, 4, 4, 6)
		A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
		test.run(tf.global_variables_initializer())
		out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
		print("out = " + str(out[0][1][1][0]))

	# Test convolutional_block part
	tf.reset_default_graph()

	with tf.Session() as test:
		np.random.seed(1)
		A_prev = tf.placeholder("float", [3, 4, 4, 6])
		X = np.random.randn(3, 4, 4, 6)
		A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
		test.run(tf.global_variables_initializer())
		out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
		print("out = " + str(out[0][1][1][0]))
	"""

	filepath = "./checkpoint/"
	filename = "saved-model-40.hdf5"

	# Run ResNets-50 part
	model = ResNet50(input_shape = (64, 64, 3), classes = 6)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	### LOAD DATASETS ###

	print ("\n=============[ LOAD DATASETS ]=============")

	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

	# Normalize image vectors
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.

	# Convert training and test labels to one hot matrices
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T

	print (" * number of training examples = " + str(X_train.shape[0]))
	print (" * number of test examples = " + str(X_test.shape[0]))
	print (" * X_train shape: " + str(X_train.shape))
	print (" * Y_train shape: " + str(Y_train.shape))
	print (" * X_test shape: " + str(X_test.shape))
	print (" * Y_test shape: " + str(Y_test.shape))
	print ("")

	if not os.path.exists(filepath + filename):

		### TRAINING PART ###
		epochs = 40
		batch_size = 64

		# checkpoint name
		filename = "saved-model-{epoch:02d}.hdf5"

		if not os.path.exists(filepath):
			os.makedirs(filepath)

		print ("\n=============[ START TRAINING ]=============")
		print (" * epochs: " + str(epochs))
		print (" * batch_size: " + str(batch_size))
		print ("")

		# set to save the checkpoint
		checkpoint = ModelCheckpoint(filepath + filename, verbose=1, save_best_only=False, mode='auto')
		callbacks_list = [checkpoint]

		# training the model
		model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, callbacks=callbacks_list, verbose=1)


	### PREDICTION PART ###

	print ("\n=============[ START PREDICTION ]=============")

	# load model weights
	model.load_weights(filepath + filename)

	preds = model.evaluate(X_test, Y_test)
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))
	print ("")


if __name__ == '__main__':
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	main()