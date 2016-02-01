#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice
from io import BytesIO

from PIL import Image
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Autoencoder
from keras.optimizers import SGD

LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 10000
TRAINING_DROPOUT_RATE = 0.8
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 64
BATCH_SIZE = 1
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

# Create model
encoder = Sequential()
encoder.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)))
encoder.add(Activation('relu'))
encoder.add(Convolution2D(32, 3, 3))
encoder.add(Activation('relu'))
encoder.add(MaxPooling2D(pool_size=(2, 2)))
encoder.add(Dropout(0.25))
encoder.add(Convolution2D(64, 3, 3, border_mode='valid'))
encoder.add(Activation('relu'))
encoder.add(Convolution2D(64, 3, 3))
encoder.add(Activation('relu'))
encoder.add(MaxPooling2D(pool_size=(2, 2))
encoder.add(Dropout(0.25))
encoder.add(Flatten())
encoder.add(Dense(256))
encoder.add(Activation('relu'))
encoder.add(Dropout(0.5))
encoder.add(Dense(10)
encoder.add(Activation('softmax'))

decoder = Sequential()
decoder.add(Dense(32, input_dim=10))
decoder.add(Activation('relu'))
decoder.add(Dropout(0.25))
decoder.add(Dense(output_dim=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)))

autoencoder = Sequential()
autoencoder.add(Autoencoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
# do autoencoder.compile(optimizer='sgd', loss='mse') after changing output_reconstruction.
autoencoder.compile(loss='msle', optimizer='adagrad')

# Define objects
input_batch = K.placeholder(ndim=3) # or shape=(None, 2, 3, 5)

# Define data-source iterator
def gather_batch(file_glob, batch_size):
	reader = tf.WholeFileReader()
	filenames = glob(file_glob)
	while True:
		batch = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
		num_samples = 0
		while num_samples < batch_size:
			try:
				filename = choice(filenames)
				img = Image.open(filename)
				print("Loaded image {}".format(filename))
				batch[num_samples,:,:,:] = np.asarray(img, dtype=np.float)/255.0
				num_samples += 1
			except ValueError as e:
				print("Problem loading image {}: {}".format(filename, e))
				continue
		yield batch
			
# Run!
# Spin up data iterator.
generator = gather_batch(sys.argv[1], BATCH_SIZE)

datagen = ImageDataGenerator(featurewise_center=True, rotation_range=20, width_shift_range=0.2, height_shigt_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

