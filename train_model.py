#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice, randint
from io import BytesIO
from itertools import cycle
import time

from PIL import Image
import numpy as np
import tensorflow as tf

# Display for debugging
np.set_printoptions(suppress=True, precision=25, linewidth=200)

LEARNING_RATE = 0.0001
TRAINING_ITERATIONS = 5000000
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 1000
BATCH_SIZE = 1
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 3

def xavier_init(shape, constant=1):
	val = constant * np.sqrt(2.0/float(np.sum(np.abs(shape[1:]))))
	return tf.random_uniform(shape, minval=-val, maxval=val)

def build_fc(input_source, hidden_size, weight=None, bias=None, activate=True):
	# Figure out size of input and create weight matrix of appropriate size.
	shape = input_source.get_shape().as_list()[-1]
	if weight is None:
		weight = tf.Variable(xavier_init([shape, hidden_size])) #tf.random_normal([shape, hidden_size]))
	if bias is None:
		bias = tf.Variable(tf.zeros([hidden_size,]))

	# Get preactivations
	result = tf.nn.bias_add(tf.matmul(input_source, weight), bias)

	# Sometimes activate
	if activate:
		result = tf.nn.tanh(result)

	return result, weight, bias

def build_conv(source, filter_shape, strides, padding='SAME', activate=True):
	we = tf.Variable(xavier_init(filter_shape))
	be = tf.Variable(tf.zeros([filter_shape[-1],]))
	conv = tf.nn.bias_add(tf.nn.conv2d(source, filter=we, strides=strides, padding=padding), be)
	if activate:
		act = tf.nn.tanh(conv) # Not relu6
	else:
		act = conv
	return act, we, be

def build_deconv(source, output_shape, filter_shape, strides, padding='SAME', activate=True):
	wd = tf.Variable(xavier_init(filter_shape))
	bd = tf.Variable(tf.zeros(output_shape[1:]))
	deconv = tf.nn.conv2d_transpose(source, filter=wd, strides=strides, padding=padding, output_shape=output_shape)
	if activate:
		act = tf.nn.tanh(deconv)
	else:
		act = deconv
	return act, wd, bd

def build_max_pool(source, kernel_shape, strides):
	return tf.nn.max_pool(source, ksize=kernel_shape, strides=strides, padding='SAME')

def build_unpool(source, kernel_shape):
	input_shape = source.get_shape().as_list()
	return tf.image.resize_images(source, input_shape[1]*kernel_shape[1], input_shape[2]*kernel_shape[2])

def build_dropout(source, toggle):
	return tf.nn.dropout(source, toggle)

def build_lrn(source):
	return tf.nn.local_response_normalization(source)

# Create model
def build_model(image_input_source, encoder_input_source, dropout_toggle):
	"""Image and Encoded are input placeholders.  input_encoded_interp is the toggle between input (when 0) and encoded (when 1).
	Returns a decoder and the encoder output."""
	# We have to match this output size.
	batch, input_height, input_width, input_depth = image_input_source.get_shape().as_list()

	# Convolutional ops will go here.
	c0, wc0, bc0 = build_conv(image_input_source, [3, 3, 3, 256], [1, 2, 2, 1], activate=False)
	c1 = build_max_pool(c0, [1, 2, 2, 1], [1, 2, 2, 1])
	c2, wc2, bc2 = build_conv(c1, [3, 3, 256, 32], [1, 1, 1, 1])
	c3 = build_max_pool(c2, [1, 2, 2, 1], [1, 2, 2, 1])
	c4, wc4, bc4 = build_conv(build_dropout(c3, dropout_toggle), [3, 3, 32, 64], [1, 1, 1, 1])
	c5 = build_max_pool(c4, [1, 2, 2, 1], [1, 2, 2, 1])
	c6, wc6, bc6 = build_conv(build_dropout(c5, dropout_toggle), [3, 3, 64, 128], [1, 1, 1, 1])
	c7 = build_max_pool(c6, [1, 2, 2, 1], [1, 2, 2, 1])
	c8, wc8, bc8 = build_conv(c7, [3, 3, 128, 256], [1, 1, 1, 1])
	c9 = build_max_pool(c8, [1, 2, 2, 1], [1, 2, 2, 1])
	conv_output = c9

	# Transition to FC layers.
	pre_flat_shape = conv_output.get_shape().as_list()
	flatten = tf.reshape(conv_output, [-1, pre_flat_shape[1]*pre_flat_shape[2]*pre_flat_shape[3]])

	# Dense connections
	fc0, wf0, bf0 = build_fc(flatten, 512)
	fc1, wf1, bf1 = build_fc(build_dropout(fc0, dropout_toggle), REPRESENTATION_SIZE)

	# Output point and our encoder mix-in.
	encoded_output = fc1 #tf.nn.softmax(fc1)
	encoded_input = build_dropout(encoder_input_source + encoded_output, dropout_toggle) # Mix input and enc.
	encoded_input.set_shape(encoded_output.get_shape()) # Otherwise we can't ascertain the size.

	# More dense connections on the offset.
	fc2, wf2, bf2 = build_fc(encoded_input, 512)
	fc3, wf3, bf3 = build_fc(build_dropout(fc2, dropout_toggle), flatten.get_shape().as_list()[-1])

	# Expand for more convolutional operations.
	unflatten = tf.reshape(fc3, [-1, pre_flat_shape[1], pre_flat_shape[2], pre_flat_shape[3]]) #pre_flat_shape)

	# More convolutions here.
	dc9 = build_unpool(unflatten, [1, 2, 2, 1])
	dc8, wdc8, bdc8 = build_deconv(dc9, c7.get_shape().as_list(), [3, 3, 128, 256], [1, 1, 1, 1])
	dc7 = build_unpool(dc8, [1, 2, 2, 1])
	dc6, wdc6, bdc6 = build_deconv(dc7, c5.get_shape().as_list(), [3, 3, 64, 128], [1, 1, 1, 1])
	dc5 = build_unpool(dc6, [1, 2, 2, 1])
	dc4, wdc4, bdc4 = build_deconv(build_dropout(dc5, dropout_toggle), c3.get_shape().as_list(), [3, 3, 32, 64], [1, 1, 1, 1])
	dc3 = build_unpool(dc4, [1, 2, 2, 1])
	dc2, wdc2, bdc2 = build_deconv(build_dropout(dc3, dropout_toggle), c1.get_shape().as_list(), [3, 3, 256, 32], [1, 1, 1, 1])
	dc1 = build_unpool(dc2, [1, 2, 2, 1])
	dc0, wdc7, bdc7 = build_deconv(dc1, [batch, input_height, input_width, input_depth], [3, 3, 3, 256], [1, 2, 2, 1], activate=False)
	deconv_output = dc0

	# Return result + encoder output
	return deconv_output, encoded_output


# Define data-source iterator
def example_generator(file_glob, noise=0.1, cache=True):
	filenames = glob(file_glob)
	file_cache = dict()
	#for filename in cycle(filenames):
	while True:
		filename = choice(filenames)
		example = None
		target = None
		if cache and filename in file_cache:
			target = file_cache[filename]
		else:
			try:
				filename = choice(filenames)
				img = Image.open(filename)
				print("Loaded image {}".format(filename))
				# Shrink image and embed in the middle of our target data.
				target_width, target_height = img.size
				max_dim = max(img.size)
				new_width = (IMAGE_WIDTH*img.size[0])//max_dim
				new_height = (IMAGE_HEIGHT*img.size[1])//max_dim
				# Center image in new image.
				newimg = Image.new(img.mode, (IMAGE_HEIGHT, IMAGE_WIDTH))
				offset_x = int((IMAGE_WIDTH/2)-(new_width/2))
				offset_y = int((IMAGE_HEIGHT/2)-(new_height/2))
				box = (offset_x, offset_y, offset_x+new_width, offset_y+new_height)
				newimg.paste(img.resize((new_width, new_height)), box)
				# Copy to target
				target = np.asarray(newimg, dtype=np.float)/255.0
				#example = np.swapaxes(example, 1, 2)
				file_cache[filename] = target
			except ValueError as e:
				print("Problem loading image {}: {}".format(filename, e))
				continue
		# Add noise
		if noise > 0:
			# Example is the noised copy.
			example = target + np.random.uniform(low=-noise, high=+noise, size=target.shape)
		else:
			example = target
		yield example, target


# Define objects
input_batch = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name="image_input")
encoded_batch = tf.placeholder(tf.float32, [BATCH_SIZE, REPRESENTATION_SIZE], name="encoder_input") # Replace BATCH_SIZE with None
keep_prob = tf.placeholder(tf.float32, name="keep_probability")
output_objective = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name="output_objective")

# Define the batch iterator
gen = example_generator(sys.argv[1], noise=0.1)
def get_batch(batch_size):
	batch = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
	labels = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
	index = 0
	while index < batch_size:
		#for index, data in enumerate(gen):
		data = next(gen)
		x, y = data
		batch[index,:,:,:] = x[:,:,:]
		labels[index,:,:,:] = y[:,:,:]
		#if index >= batch_size:
		#	break
		index += 1
	return batch, labels


# Convenience method for writing an output image from an encoded array
def save_image(struct, filename):
	#img_tensor = tf.image.encode_jpeg(decoded[0])
	decoded_min = struct[0].min()
	decoded_max = struct[0].max()
	decoded_norm = (struct[0]-decoded_min)/(decoded_max-decoded_min)
	img_arr = np.asarray(decoded_norm*255, dtype=np.uint8)
	img = Image.fromarray(img_arr)
	img.save(filename)

def save_reconstruction(session, decoder, array, filename):
	decoded = session.run(decoder, feed_dict={
		input_batch:np.zeros(shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), 
		encoded_batch:array, 
		keep_prob:1.0,
	})
	save_image(decoded, filename)


# Run!
with tf.Session() as sess:
	# Populate autoencoder in session and gather pretrainers.
	decoder, encoder = build_model(input_batch, encoded_batch, keep_prob)
	# Get final ops
	global_reconstruction_loss = tf.reduce_sum(np.abs(output_objective - decoder))
	#global_reconstruction_loss = tf.reduce_sum((output_objective - decoder)**2)
	#global_reconstruction_loss = tf.nn.l2_loss(output_objective - decoder)
	global_representation_loss = tf.abs(1-tf.reduce_sum(encoder)) + tf.reduce_sum(tf.abs(encoder))
	global_loss = global_reconstruction_loss + global_representation_loss
	#global_optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(global_loss) #tf.clip_by_value(global_loss, -1e6, 1e6))
	global_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(global_loss)

	# Init variables.
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())

	# If we already have a trained network, reload that. The saver doesn't save the graph structure, so we need to build one with identical node names and reload it here.
	# Right now the graph can be loaded with tf.import_graph_def OR the variables can be populated with Restore, but not both (yet).
	# The graph.pbtxt holds graph structure (in model folder).  model-checkpoint has values/weights.
	# TODO: Review when bug is fixed. (2015/11/29)
	if os.path.isfile("./model/checkpoint"):
		print("Restored model state.")
		saver.restore(sess, "./model/checkpoint.model")
	else:
		print("No model found.  Starting new model.")

	# Begin training
	for iteration in range(1, TRAINING_ITERATIONS):
		try:
			x_batch, y_batch = get_batch(BATCH_SIZE)
			loss1, _, encoder_output = sess.run(
				[global_loss, global_optimizer, encoder], 
				feed_dict={
					input_batch:x_batch, 
					encoded_batch:np.zeros((BATCH_SIZE, REPRESENTATION_SIZE)),
					output_objective:y_batch,
					keep_prob:0.5,
				}
			) # y_batch is denoised.
			print("Iter {}: {} \n {} \n {}".format(iteration, loss1, encoder_output.sum(), encoder_output[0,:]))
			if iteration % TRAINING_REPORT_INTERVAL == 0:
				# Checkpoint progress
				print("Finished batch {}".format(iteration))
				saver.save(sess, "./model/checkpoint.model") #, global_step=iteration)

				# Render output sample
				encoded = sess.run(encoder, feed_dict={
					input_batch:y_batch, 
					keep_prob:1.0,
				})

				# Randomly generated sample
				#decoded = sess.run(decoder, feed_dict={encoded_batch:np.random.normal(loc=encoded.mean(), scale=encoded.std(), size=[BATCH_SIZE, REPRESENTATION_SIZE])})
				print("Encoded: {}".format(encoded))
				save_reconstruction(sess, decoder, encoded, "test_{:08d}.jpg".format(iteration))
				time.sleep(1.0) # Sleep to avoid shared process killing us for resources.
		except KeyboardInterrupt:
			from IPython.core.debugger import Tracer
			Tracer()()
	
	# When complete, stop and let us play.
	print("Finished")
	for i in range(REPRESENTATION_SIZE):
		encoded = np.zeros((BATCH_SIZE, REPRESENTATION_SIZE))
		encoded[0,i] = 1.0
		save_reconstruction(sess, decoder, encoded, "enc_{}.jpg".format(i))
	from IPython.core.debugger import Tracer
	Tracer()()
