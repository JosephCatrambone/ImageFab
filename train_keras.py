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
from keras.models import Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

MODEL_JSON = "model.json"
MODEL_WEIGHTS = "model.h5"

TRAINING_ITERATIONS = 5
REPRESENTATION_SIZE = 64
BATCH_SIZE = 10
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

SHAPE_ORDERING = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH) # move depth to last for TensorFlow backend

# Define objects
#input_batch = K.placeholder(shape=SHAPE_ORDERING) #ndim=3) or shape=(None, 2, 3, 5)

def save_model(model, json_filename=MODEL_JSON, weight_filename=MODEL_WEIGHTS):
	fout = open(json_filename, 'w')
	fout.write(model.to_json())
	fout.close()
	model.save_weights(weight_filename, overwrite=True)
	
def load_model(json_filename=MODEL_JSON, weight_filename=MODEL_WEIGHTS):
	if not os.path.isfile(json_filename) or not os.path.isfile(weight_filename):
		return None
	fin = open(json_filename, 'r')
	model = model_from_json(fin.read())
	fin.close()
	model.load_weights(weight_filename)
	return model

def build_model(image_input_name='image_input', representation_input_name='representation_input'):
	graph = Graph()
	graph.add_input(name=representation_input_name, input_shape=(REPRESENTATION_SIZE,))
	graph.add_input(name=image_input_name, input_shape=SHAPE_ORDERING)

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

	graph.add_node(Dense(512, input_dim=10), name='op19', inputs=['op18', 'representation_input'], merge_mode='ave')
	graph.add_node(Activation('relu'), name='op20', input='op19')
	graph.add_node(Dropout(0.25), name='op21', input='op20')
	graph.add_node(Dense(IMAGE_DEPTH*IMAGE_HEIGHT*IMAGE_WIDTH), name='op22', input='op21')
	graph.add_node(Reshape(SHAPE_ORDERING), name='op23', input='op22')

	#graph.add_output(name='encoded_output', input='op18')
	graph.add_output(name='decoded_output', input='op23')

	graph.compile(optimizer='rmsprop', loss={'decoded_output':'msle'})

	return graph

#history = graph.fit({'image_input':batch, 'decoded_output':batch}, nb_epoch=10)
#prediction = graph.predict({'representation_input':rep})

# Define data-source iterator
def example_generator(file_glob, noise=0.0):
	filenames = glob(file_glob)
	for filename in cycle(filenames):
		example = None
		try:
			filename = choice(filenames)
			img = Image.open(filename)
			print("Loaded image {}".format(filename))
			example = np.asarray(img, dtype=np.float)/255.0
			# If tensorflow backend, H, W, D
			# If theano, D, H, W
			example = np.swapaxes(example, 1, 2)
			example = np.swapaxes(example, 0, 1)
			if noise > 0:
				example += np.random.uniform(low=-noise, high=+noise, shape=example.shape)
		except ValueError as e:
			print("Problem loading image {}: {}".format(filename, e))
			continue
		yield example
			
# Run!
if __name__=="__main__":
	model = load_model()
	if model:
		print("Loaded model.  Resuming training.")
	else:
		print("Model not loaded.  Starting from scratch.")
		model = build_model()
	generator = example_generator(sys.argv[1])
	while True:
		# Fill out a bunch of training examples
		X_set = np.zeros([BATCH_SIZE] + list(SHAPE_ORDERING), dtype=np.float)
		for i, example in zip(range(BATCH_SIZE), generator):
			X_set[i,:,:,:] = example
		# Fit our model
		model.fit({'image_input':X_set, 'representation_input':np.zeros((BATCH_SIZE, REPRESENTATION_SIZE), dtype=np.float), 'decoded_output':X_set}, nb_epoch=TRAINING_ITERATIONS)

		# Generate an example
		pred = model.predict({'image_input':np.zeros(([BATCH_SIZE] + list(SHAPE_ORDERING)), dtype=np.float), 'representation_input':np.random.uniform(size=(BATCH_SIZE, REPRESENTATION_SIZE))})

		# Write a sample image
		image_data = pred['decoded_output'][0,:,:,:] # 3, 255, 255
		image_data = np.swapaxes(image_data, 0, 1) # 255, 3, 255
		image_data = np.swapaxes(image_data, 1, 2) # 255, 255, 3
		image_data -= image_data.min()
		image_data /= (image_data.max() + 1.0e-6)
		img = Image.fromarray(np.asarray(image_data * 255, dtype=np.uint8))
		img.save("sample.png")

		save_model(model)

