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

LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 50000
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 50
BATCH_SIZE = 1
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3

def build_fc(hidden_size, input_source, weight=None, bias=None, activate=True):
	shape = input_source.get_shape().as_list()[-1]
	if weight is None:
		weight = tf.Variable(tf.random_normal([shape, hidden_size]))
	if bias is None:
		bias = tf.Variable(tf.random_normal([hidden_size,]))

	result = tf.nn.bias_add(tf.matmul(input_source, weight), bias)

	if activate:
		result = tf.nn.tanh(result)

	return result, weight, bias

# Create model
def build_model(image_input_source, encoder_input_source, input_encoder_interpolation, dropout_toggle):
	"""Image and Encoded are input placeholders.  input_encoded_interp is the toggle between input (when 0) and encoded (when 1).
	Returns a decoder and the encoder output."""

	batch, input_height, input_width, input_depth = image_input_source.get_shape().as_list()

	flatten = tf.reshape(image_input_source, [-1, input_height*input_width*input_depth])

	fc0, w0, b0 = build_fc(128, flatten, activate=False)
	fc1, w1, b1 = build_fc(REPRESENTATION_SIZE, fc0)

	encoded_output = fc1
	encoded_input = encoder_input_source*input_encoder_interpolation + (1-input_encoder_interpolation)*encoded_output
	encoded_input.set_shape(encoded_output.get_shape()) # Otherwise we can't ascertain the size.

	fc2, w2, b2 = build_fc(128, encoded_input)
	fc3, w3, b3 = build_fc(input_height*input_width*input_depth, fc2, activate=False)

	unflatten = tf.reshape(fc3, image_input_source.get_shape().as_list()) #[-1, input_height, input_width, input_depth])

	return unflatten, encoded_output

# Other methods we need to finish migrating.
def add_softmax(self):
	self._add_softmax_encoder(self._last_encoder)
	self._last_encoder = self.encoder_operations[-1]

	encoder = self.encoder_operations[-1]
	def anon(stream_signal):
		self._add_softmax_decoder(encoder, stream_signal)
	self.build_queue.append(anon)

def _add_softmax_encoder(self, input_to_encode):
	print("SOFTMAX ENC")
	act = tf.nn.softmax(input_to_encode)

	self.encoder_operations.append(act)
	self.encoder_weights.append(None)
	self.encoder_biases.append(None)

def _add_softmax_decoder(self, signal_from_encoder, input_to_decode):
	print("SOFTMAX DEC")
	# Decode requires two steps.  First, decoder path.
	dec = tf.nn.softmax(input_to_decode)

	self.decoder_operations.append(dec)
	self.decoder_weights.append(None)
	self.decoder_biases.append(None)

	# Second, autoencoder path.
	ae = tf.nn.softmax(signal_from_encoder)
	self.pretrainer_operations.append(ae)

def add_local_response_normalization(self):
	self._add_lrn_encoder(self._last_encoder)
	self._last_encoder = self.encoder_operations[-1]

	encoder = self.encoder_operations[-1]
	def anon(backward_stream):
		self._add_lrn_decoder(encoder, backward_stream)
	self.build_queue.append(anon)

def _add_lrn_encoder(self, input_to_encode):
	enc_op = tf.nn.local_response_normalization(input_to_encode)
	self.encoder_operations.append(enc_op)
	self.encoder_weights.append(None)
	self.encoder_biases.append(None)

def _add_lrn_decoder(self, signal_from_encoder, input_to_decode):
	# Decoder path is straight.
	ident = tf.identity(input_to_decode)

	self.decoder_operations.append(ident)
	self.decoder_weights.append(None)
	self.decoder_biases.append(None)

	# And decoder
	ident2 = tf.identity(signal_from_encoder)
	self.pretrainer_operations.append(ident2)

def add_conv2d(self, filter_height, filter_width, filter_depth, num_filters, strides=None, padding='SAME', activate=True):
	if not strides:
		strides = [1, filter_height, filter_width, 1]
	input_size = self._last_encoder.get_shape().as_list()
	filter_shape = [filter_height, filter_width, filter_depth, num_filters]

	self._add_conv_encoder(self._last_encoder, filter_shape, strides, padding, activate)
	self._last_encoder = self.encoder_operations[-1]

	encoder_ref = self.encoder_operations[-1]
	def anon(signal_to_decode):
		self._add_conv_decoder(encoder_ref, signal_to_decode, input_size, filter_shape, strides, padding, activate)
	self.build_queue.append(anon)

def _add_conv_encoder(self, input_to_encode, filter_shape, strides, padding='SAME', activate=True):
	print("CONV ENC {} (x) {}".format(input_to_encode.get_shape(), filter_shape))
	# Encode phase
	we = tf.Variable(tf.random_normal(filter_shape))
	be = tf.Variable(tf.random_normal([filter_shape[-1],]))
	conv = tf.nn.conv2d(input_to_encode, filter=we, strides=strides, padding=padding) + be
	if activate:
		act1 = tf.nn.relu6(conv)
	else:
		act1 = conv
	#pool = tf.nn.max_pool(act1, ksize=[1, filter_shape[1], filter_shape[2], 1], strides=[1, filter_shape[1], filter_shape[2], 1], padding='SAME')
	#norm = tf.nn.lrn(pool, strides[1], bias=1.0, alpha=0.001, beta=0.75)

	self.encoder_operations.append(act1)
	self.encoder_weights.append(we)
	self.encoder_biases.append(be)

def _add_conv_decoder(self, signal_from_encoder, input_to_decode, input_size, filter_size, strides, padding='SAME', activate=True):
	print("CONV DEC {} (x) {}".format(input_size, filter_size))
	# Decode phase
	dec_shape = signal_from_encoder.get_shape().as_list()

	# Deconv2D args:
	wd = tf.Variable(tf.random_normal(filter_size))
	bd = tf.Variable(tf.random_normal([input_size[1], input_size[2], input_size[3],]))
	deconv = tf.nn.conv2d_transpose(input_to_decode, filter=wd, strides=strides, padding=padding, output_shape=input_size) + bd
	if activate:
		act = tf.nn.relu6(deconv)
	else:
		act = deconv

	self.decoder_operations.append(act)
	self.decoder_weights.append(wd)
	self.decoder_biases.append(bd)

	# Autoencode phase
	autoenc = tf.nn.conv2d_transpose(signal_from_encoder, filter=wd, strides=strides, padding=padding, output_shape=input_size) + bd
	if activate:
		ae_act = tf.nn.relu6(autoenc)
	else:
		ae_act = autoenc
	self.pretrainer_operations.append(ae_act)

def add_pool(self, batch_size, kernel_height, kernel_width, kernel_depth, strides=None):
	input_shape = self._last_encoder.get_shape().as_list()

	if strides is None:
		strides = [1, kernel_height, kernel_width, 1]

	self._add_pool_encoder(self._last_encoder, [batch_size, kernel_height, kernel_width, kernel_depth], strides)
	encoder_reference = self.encoder_operations[-1]
	self._last_encoder = encoder_reference

	def decoder_builder(signal_to_decode):
		self._add_pool_decoder(encoder_reference, signal_to_decode, input_shape, [batch_size, kernel_height, kernel_width, kernel_depth], [1, 1])
	self.build_queue.append(decoder_builder)

def _add_pool_encoder(self, input_to_encode, kernel_shape, strides):
	print("POOL ENC {} (x) {}".format(input_to_encode.get_shape(), kernel_shape))
	pool = tf.nn.max_pool(input_to_encode, ksize=kernel_shape, strides=strides, padding='SAME')

	self.encoder_operations.append(pool)
	self.encoder_weights.append(None)
	self.encoder_biases.append(None)

def _add_pool_decoder(self, signal_from_encoder, input_to_decode, input_shape, kernel_shape, strides):
	print("POOL DEC {} (x) {}".format(input_shape, kernel_shape))
	# Deconv2D args:
	deconv = tf.image.resize_images(input_to_decode, input_shape[1], input_shape[2])

	self.decoder_operations.append(deconv)
	self.decoder_weights.append(None)
	self.decoder_biases.append(None)

	# Autoencode phase
	autoenc = tf.image.resize_images(signal_from_encoder, input_shape[1], input_shape[2])
	self.pretrainer_operations.append(autoenc)

def add_dropout(self, dropout_toggle):
	self._add_dropout_encoder(self._last_encoder, dropout_toggle)
	encoder_ref = self.encoder_operations[-1]
	self._last_encoder = encoder_ref

	def anon(signal_to_decode):
		self._add_dropout_decoder(encoder_ref, signal_to_decode, dropout_toggle)
	self.build_queue.append(anon)

def _add_dropout_encoder(self, to_encode, dropout_toggle):
	print("DROPOUT ENC")
	# Encode
	drop = tf.nn.dropout(to_encode, dropout_toggle)

	self.encoder_operations.append(drop)
	self.encoder_weights.append(None)
	self.encoder_biases.append(None)

def _add_dropout_decoder(self, signal_from_encoder, input_to_decode, dropout_toggle):
	print("DROPOUT DEC")
	# Decode
	drop = tf.nn.dropout(input_to_decode, dropout_toggle)

	self.decoder_operations.append(drop)
	self.decoder_weights.append(None)
	self.decoder_biases.append(None)

	# Not strictly necessary, but...
	autoenc = tf.nn.dropout(signal_from_encoder, dropout_toggle)
	self.pretrainer_operations.append(autoenc)

def add_flatten(self):
	input_shape = self._last_encoder.get_shape().as_list()

	self._add_flatten_encoder(self._last_encoder, *input_shape)
	encoder_ref = self.encoder_operations[-1]
	self._last_encoder = encoder_ref

	def anon(signal_to_decode):
		self._add_flatten_decoder(encoder_ref, signal_to_decode, *input_shape)
	self.build_queue.append(anon)

def _add_flatten_encoder(self, to_encode, batch_size, input_height, input_width, input_depth):
	# Encode
	flatten = tf.reshape(to_encode, [-1, input_height*input_width*input_depth])

	self.encoder_operations.append(flatten)
	self.encoder_weights.append(None)
	self.encoder_biases.append(None)

def _add_flatten_decoder(self, signal_from_encoder, input_to_decode, batch_size, input_height, input_width, input_depth):
	# Decode
	unflatten = tf.reshape(input_to_decode, [-1, input_height, input_width, input_depth])

	self.decoder_operations.append(unflatten)
	self.decoder_weights.append(None)
	self.decoder_biases.append(None)

	# Not strictly necessary, but...
	autoenc = tf.reshape(signal_from_encoder, [-1, input_height, input_width, input_depth])
	self.pretrainer_operations.append(autoenc)


# Define data-source iterator
def example_generator(file_glob, noise=0, cache=True):
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
input_encoder_interpolation = tf.placeholder(tf.float32, name="input_encoder_interpolation")
keep_prob = tf.placeholder(tf.float32, name="keep_probability")


# Define the batch iterator
gen = example_generator(sys.argv[1], noise=0.0)
def get_batch(batch_size):
	batch = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
	labels = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
	for index, data in enumerate(gen):
		if index >= batch_size:
			break
		x, y = data
		batch[index,:,:,:] = x[:,:,:]
		labels[index,:,:,:] = y[:,:,:]
	return batch, labels


# Convenience method for writing an output image from an encoded array
def save_reconstruction(session, decoder, array, filename):
	decoded = session.run(decoder, feed_dict={
		input_batch:np.zeros(shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), 
		encoded_batch:array, 
		keep_prob:1.0,
		input_encoder_interpolation:1.0, # 100% encoder.
	})
	#img_tensor = tf.image.encode_jpeg(decoded[0])
	decoded_min = decoded[0].min()
	decoded_max = decoded[0].max()
	decoded_norm = (decoded[0]-decoded_min)/(decoded_max-decoded_min)
	img_arr = np.asarray(decoded_norm*255, dtype=np.uint8)
	img = Image.fromarray(img_arr)
	img.save(filename)


# Run!
with tf.Session() as sess:
	# Populate autoencoder in session and gather pretrainers.
	decoder, encoder = build_model(input_batch, encoded_batch, input_encoder_interpolation, keep_prob)
	# Get final ops
	global_reconstruction_loss = tf.nn.l2_loss(input_batch - decoder)
	#global_representation_loss = tf.pow(tf.abs(1 - np.sum(tf.abs(encoder))), SPARSITY_PENALTY)
	global_loss = global_reconstruction_loss# + global_representation_loss 
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
					keep_prob:0.5, 
					encoded_batch:np.random.uniform(low=-1e-10, high=1e-10, size=[BATCH_SIZE, REPRESENTATION_SIZE]),
					input_encoder_interpolation:0, # Purely input
				}
			) # y_batch is denoised.
			print("Iter {}: {} \t {}".format(iteration, loss1, encoder_output[0,:].sum()))
			if iteration % TRAINING_REPORT_INTERVAL == 0:
				# Checkpoint progress
				print("Finished batch {}".format(iteration))
				saver.save(sess, "./model/checkpoint.model") #, global_step=iteration)

				# Render output sample
				encoded = sess.run(encoder, feed_dict={
					input_batch:x_batch, 
					keep_prob:1.0,
					input_encoder_interpolation:0,
				})

				# Randomly generated sample
				#decoded = sess.run(decoder, feed_dict={encoded_batch:np.random.normal(loc=encoded.mean(), scale=encoded.std(), size=[BATCH_SIZE, REPRESENTATION_SIZE])})
				print("Encoded: {}".format(encoded))
				save_reconstruction(sess, decoder, encoded, "test_{}.jpg".format(iteration))
				time.sleep(1.0) # Sleep to avoid shared process killing us for resources.
		except KeyboardInterrupt:
			from IPython.core.debugger import Tracer
			Tracer()()
