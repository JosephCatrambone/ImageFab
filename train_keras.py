#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice
from itertools import cycle
from io import BytesIO

from PIL import Image
import numpy as np
from keras import backend as K
from keras.layers import containers
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

SHAPE_ORDERING = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) # move depth to last for TensorFlow backend

# Define objects
input_batch = K.placeholder(shape=SHAPE_ORDERING) #ndim=3) or shape=(None, 2, 3, 5)

# Create model
graph = Graph()
graph.add_input(name='representation_input', input_shape=(REPRESENTATION_SIZE,))
graph.add_input(name='image_input', input_shape=SHAPE_ORDERING)

graph.add_node(Convolution2D(32, 3, 3, border_mode='valid'), name='op1', input='image_input')
graph.add_node(Activation('relu'), name='op2', input='op1')
graph.add_node(Convolution2D(32, 3, 3), name='op3', input='op2')
graph.add_node(Activation('relu'), name='op4', input='op3')
graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='op5', input='op4')
graph.add_node(Dropout(0.25), name='op6', input='op5')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid'), name='op7', input='op6')
graph.add_node(Activation('relu'), name='op8', input='op7')
graph.add_node(Convolution2D(64, 3, 3), name='op9', input='op8')
graph.add_node(Activation('relu'), name='op10', input='op9')
graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='op11', input='op10')
graph.add_node(Dropout(0.25), name='op12', input='op11')
graph.add_node(Flatten(), name='op13', input='op12')
graph.add_node(Dense(256), name='op14', input='op13')
graph.add_node(Activation('relu'), name='op15', input='op14')
graph.add_node(Dropout(0.5), name='op16', input='op15')
graph.add_node(Dense(REPRESENTATION_SIZE), name='op17', input='op16')
graph.add_node(Activation('softmax'), name='op18', input='op17')

graph.add_node(Dense(32, input_dim=10), name='op19', inputs=['op18', 'representation_input'], merge_mode='ave')
graph.add_node(Activation('relu'), name='op20', input='op19')
graph.add_node(Dropout(0.25), name='op21', input='op20')
graph.add_node(Dense(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH), name='op22', input='op21')
graph.add_node(Reshape(SHAPE_ORDERING), name='op23', input='op22')

#graph.add_output(name='encoded_output', input='op18')
graph.add_output(name='decoded_output', input='op23')

graph.compile(optimizer='rmsprop', loss={'decoded_output':'msle'})

#history = graph.fit({'image_input':batch, 'decoded_output':batch}, nb_epoch=10)
#prediction = graph.predict({'representation_input':rep})

# Define data-source iterator
def loader(file_glob):
	filenames = glob(file_glob)
	for filename in cycle(filenames):
		num_samples = 0
		try:
			filename = choice(filenames)
			img = Image.open(filename)
			print("Loaded image {}".format(filename))
			batch = np.asarray(img, dtype=np.float)/255.0
			# If tensorflow backend, H, W, D
			# If theano, D, H, W
			batch = np.swapaxes(batch, 1, 2)
			batch = np.swapaxes(batch, 0, 1)
		except ValueError as e:
			print("Problem loading image {}: {}".format(filename, e))
			continue
		yield batch
			
# Run!
# Spin up data iterator.
generator = loader(sys.argv[1])
X_set = np.zeros([10] + list(SHAPE_ORDERING), dtype=np.float)
for i, example in zip(range(10), generator):
	X_set[i,:,:,:] = example
graph.fit({'image_input':X_set, 'representation_input':np.zeros((10, REPRESENTATION_SIZE), dtype=np.float), 'decoded_output':X_set}, nb_epoch=10)
#datagen = ImageDataGenerator(featurewise_center=True, rotation_range=20, width_shift_range=0.2, height_shigt_range=0.2, horizontal_flip=True)
#datagen.fit(X_train)
pred = graph.predict({'representation_input':np.random.uniform(size=(10, REPRESENTATION_SIZE))})

