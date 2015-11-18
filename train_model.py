#!/usr/bin/env python
import sys, os

import numpy as np
import tensorflow as tf

REPRESENTATION_SIZE = 64
BATCH_SIZE = 5
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

def build_encoder(image_batch, keep_prob, representation_size=REPRESENTATION_SIZE):
	# Conv -> Bias -> Pool -> Norm -> Dropout

	# Conv 1
	cw1 = tf.Variable(tf.random_normal([5, 5, batch_shape[3].value, 256]))
	cb1 = tf.Variable(tf.random_normal([256,]))
	conv1 = tf.nn.conv2d(image_batch, filter=cw1, strides=[1, 1, 1, 1], padding='SAME')
	biased1 = tf.nn.bias_add(conv1, cb1) # Special case of +cb1 which is a 1D-Tensor cast.
	act1 = tf.nn.relu(biased1)
	pool1 = tf.nn.max_pool(act1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')
	norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	drop1 = tf.nn.dropout(norm1, keep_prob)
	
	# Conv 2
	cw2 = tf.Variable(tf.random_normal([5, 5, 1, 256]))
	cb2 = tf.Variable(tf.random_normal([256,]))
	conv2 = tf.nn.conv2d(drop1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME')
	biased2 = tf.nn.bias_add(conv2, cb2)
	act2 = tf.nn.relu(biased2)
	pool2 = tf.nn.max_pool(act2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')
	norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001, beta=0.75)
	drop2 = tf.nn.dropout(norm2, keep_prob)

	# Record old shape
	drop_shape = drop.get_shape()
	drop_length = drop_shape[1].value*drop_shape[2].value*drop_shape[3].value

	# Reshape
	resh1 = tf.reshape(drop2, [-1, drop_length]]) # Make flat

	# FC 1
	wf1 = tf.Variable(tf.random_normal([drop_length, representation_size]))
	fb1 = tf.Variable(tf.random_normal([representation_size,])
	full1 = tf.matmul(drop1, wf1) + fb1
	act3 = tf.nn.relu(full1)

	return act3, [cw1, cw2, wf1], [cb1, cb2, fb1]

def build_decoder(representation_batch, keep_prob, output_shape)
	# FC 2
	wf2 = tf.Variable(tf.random_normal([
		representation_batch.get_shape()[1].value, 
		output_shape[1].value*output_shape[2].value*output_shape[3].value
	])
	fb2 = tf.Variable(tf.random_normal([output_shape[1].value*output_shape[2].value*output_shape[3].value,])
	full2 = tf.matmul(representation_batch, wf2) + fb2
	act4 = tf.nn.relu(full2)

	# Conv 3
	

	# Conv 4

	return 

input_batch = tf.placeholder(tf.types.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
keep_prob = tf.placeholder(tf.types.float32)
encoder = build_encoder(input_batch, keep_prob)
