#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice, randint
from io import BytesIO
from itertools import cycle

from PIL import Image
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 50000
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 64
BATCH_SIZE = 10
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 3

# Create model
class ConvolutionalAutoencoder(object):
	def __init__(self, to_encode, to_decode):
		self.to_encode = to_encode
		self.to_decode = to_decode

		self.encoder_operations = list()
		self.decoder_operations = list()
		self.pretrainer_operations = list() # Used for layer-wise pretraining. pretrainer_operations[0] is encoder -> hidden state -> decoder. 
		self.encoder_weights = list()
		self.encoder_biases = list()
		self.decoder_weights = list()
		self.decoder_biases = list()

		self.encoder_operations.append(to_encode)
		self.decoder_operations.append(to_decode)

		# Build queue stores a list of anonymous functions which accept the encoder_signal and the decoder_input.
		# After finalization, each of the functions is called in reverse order to build the decoder stream.
		# Then the build_queue is destroyed.
		# Stores a list of tuple of (type_name, args)
		self._last_encoder = self.to_encode
		self.build_queue = list() 

	def add_fc(self, hidden_size, activate=True):
		visible_size = self._last_encoder.get_shape().as_list()[-1]
		self._add_fc_encoder(self._last_encoder, visible_size, hidden_size, activate)
		self._last_encoder = self.encoder_operations[-1]

		encoder = self.encoder_operations[-1]
		def anon(stream_signal):
			self._add_fc_decoder(encoder, stream_signal, visible_size, hidden_size, activate)
		self.build_queue.append(anon)

	def _add_fc_encoder(self, input_to_encode, visible_size, hidden_size, activate=True):
		print("FC ENC {} -> {}".format(visible_size, hidden_size))
		# Encode is straightforward.  Data always comes in the same way.
		we = tf.Variable(tf.random_normal([visible_size, hidden_size]))
		be = tf.Variable(tf.random_normal([hidden_size,]))
		fc1 = tf.matmul(input_to_encode, we) + be
		if activate:
			act1 = tf.nn.relu6(fc1)
		else:
			act1 = fc1

		self.encoder_operations.append(act1)
		self.encoder_weights.append(we)
		self.encoder_biases.append(be)

	def _add_fc_decoder(self, signal_from_encoder, input_to_decode, visible_size, hidden_size, activate=True):
		print("FC DEC {} -> {}".format(hidden_size, visible_size))
		# Decode requires two steps.  First, decoder path.
		wd = tf.Variable(tf.random_normal([hidden_size, visible_size]))
		bd = tf.Variable(tf.random_normal([visible_size, ]))
		fc2 = tf.matmul(input_to_decode, wd) + bd
		if activate:
			act2 = tf.nn.relu6(fc2)
		else:
			act2 = fc2

		self.decoder_operations.append(act2)
		self.decoder_weights.append(wd)
		self.decoder_biases.append(bd)

		# Second, autoencoder path.
		fc3 = tf.matmul(signal_from_encoder, wd) + bd
		if activate:
			act3 = tf.nn.relu6(fc3)
		else:
			act3 = fc3
		self.pretrainer_operations.append(act3)

	def add_local_response_normalization(self):
		self._add_lrn_encoder(self._last_encoder)

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

	def get_layer_count(self):
		return len(self.encoder_operations)

	def get_output_shape(self):
		return self.encoder_operations[-1].get_shape()

	def get_encoder_output(self, layer=-1):
		return self.encoder_operations[layer]

	def get_decoder_output(self, layer=0):
		# NOTE: This corresponds to the output of encoder [layer], so if we decode in order from the top,
		# we'll have to run it through decoder_operations in reverse.
		# When being build, it is in the 'correct' order for reconstruction, but we flip it after build
		# to make it match up (since the graph is in the right order anyway).
		return self.decoder_operations[layer]

	def get_pretrainer_output(self, layer):
		# Similar to get decoder_output, but uses a short-circuited path, rather than the top-most decoder stream.
		return self.pretrainer_operations[layer]

	def finalize(self):
		last_decoder = self.to_decode + self._last_encoder
		for op in reversed(self.build_queue):
			op(last_decoder)
			last_decoder = self.decoder_operations[-1]

		# We appended things from the top-leve to the bottom, so decoder[n] corresponds to encoder[0].
		# Flip all the fields so they match.  
		self.decoder_operations.reverse()
		self.decoder_weights.reverse()
		self.decoder_biases.reverse()
		self.pretrainer_operations.reverse()

		self.build_queue = None
		self._last_encoder = None
		del self.build_queue
		del self._last_encoder

# Define objects
input_batch = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name="image_input")
encoded_batch = tf.placeholder(tf.float32, [BATCH_SIZE, REPRESENTATION_SIZE], name="encoder_input") # Replace BATCH_SIZE with None
keep_prob = tf.placeholder(tf.float32)
autoencoder = ConvolutionalAutoencoder(input_batch, encoded_batch)

# Define data-source iterator
def example_generator(file_glob, noise=0.0, cache=True):
	filenames = glob(file_glob)
	file_cache = dict()
	for filename in cycle(filenames):
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

gen = example_generator(sys.argv[1], noise=0.1)
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
			
# Run!
with tf.Session() as sess:
	# Populate autoencoder in session and gather pretrainers.
	autoencoder.add_conv2d(11, 11, IMAGE_DEPTH, 64, strides=[1, 5, 5, 1], activate=False)
	autoencoder.add_pool(1, 2, 2, 1, strides=[1, 1, 1, 1])
	autoencoder.add_local_response_normalization()
	autoencoder.add_dropout(keep_prob)
	autoencoder.add_conv2d(5, 5, 64, 128, strides=[1, 3, 3, 1])
	autoencoder.add_pool(1, 2, 2, 1, strides=[1, 1, 1, 1])
	autoencoder.add_local_response_normalization()
	autoencoder.add_dropout(keep_prob)
	autoencoder.add_conv2d(5, 5, 128, 256, strides=[1, 3, 3, 1])
	autoencoder.add_local_response_normalization()
	autoencoder.add_dropout(keep_prob)
	autoencoder.add_flatten()
	autoencoder.add_fc(128)
	autoencoder.add_fc(REPRESENTATION_SIZE)
	autoencoder.finalize()

	# Collect trainers.
	optimizers = list()
	for layer in range(autoencoder.get_layer_count()-1):
		enc = autoencoder.get_encoder_output(layer)
		dec = autoencoder.get_pretrainer_output(layer)
		#l2_cost = tf.reduce_sum(tf.pow(enc - dec, 2))
		l2_cost = tf.nn.l2_loss(enc - dec)
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(l2_cost)
		optimizers.append(optimizer)

	# Get final ops
	encoder = autoencoder.get_encoder_output()
	decoder = autoencoder.get_decoder_output()
	global_loss = tf.nn.l2_loss(input_batch - decoder) + tf.reduce_sum(encoder) # for reduce sum, use (encoder, 1) to get it for each example.
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
	for level, optimizer in enumerate(optimizers):
		try:
			for iteration in range(TRAINING_ITERATIONS):
				x_batch, y_batch = get_batch(BATCH_SIZE)
				#sess.run(optimizer, feed_dict={input_batch:x_batch, keep_prob:0.5, encoded_batch:np.random.uniform(low=-0.001, high=0.001, size=[BATCH_SIZE, REPRESENTATION_SIZE])})
				loss1, _ = sess.run([global_loss, global_optimizer], feed_dict={input_batch:x_batch, keep_prob:0.5, encoded_batch:np.random.uniform(low=-0.1, high=0.1, size=[BATCH_SIZE, REPRESENTATION_SIZE])}) # y_batch is denoised.
				print("Iter {}: {}".format(iteration, loss1))
				if iteration % TRAINING_REPORT_INTERVAL == 0:
					# Checkpoint progress
					print("Finished batch {}".format(iteration))
					saver.save(sess, "./model/checkpoint.model") #, global_step=iteration)

					# Render output sample
					#encoded, decoded = sess.run([encoder, decoder], feed_dict={input_batch:x_batch, encoded_batch:np.random.uniform(size=(BATCH_SIZE, REPRESENTATION_SIZE))})
					encoded = sess.run(encoder, feed_dict={input_batch:x_batch, keep_prob:1.0})

					# Randomly generated sample
					#decoded = sess.run(decoder, feed_dict={encoded_batch:np.random.normal(loc=encoded.mean(), scale=encoded.std(), size=[BATCH_SIZE, REPRESENTATION_SIZE])})
					feature = np.zeros((BATCH_SIZE, REPRESENTATION_SIZE), dtype=np.float)
					feature[0,randint(0, REPRESENTATION_SIZE-1)] = 1.0
					decoded = sess.run(decoder, feed_dict={
						input_batch:np.zeros(shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), 
						#encoded_batch:encoded, 
						encoded_batch:feature,
						#encoded_batch:np.random.uniform(low=-1.0, high=1.0, size=[BATCH_SIZE, REPRESENTATION_SIZE]), 
						keep_prob:1.0})
					#img_tensor = tf.image.encode_jpeg(decoded[0])
					decoded_prefilter = decoded/decoded.std()
					decoded_min = decoded_prefilter[0].min()
					decoded_max = decoded_prefilter[0].max()
					decoded_norm = (decoded_prefilter[0]-decoded_min)/(decoded_max-decoded_min)
					img_arr = np.asarray(decoded_norm*255, dtype=np.uint8)
					img = Image.fromarray(img_arr)
					img.save("test_{}_{}.jpg".format(level, iteration))

					# Reconstructed sample ends up looking just like the random sample, so don't waste time making it.
		except KeyboardInterrupt:
			from IPython.core.debugger import Tracer
			Tracer()()
