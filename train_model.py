#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice
from io import BytesIO

from PIL import Image
import numpy as np
import tensorflow as tf

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

	def add_fc(self, hidden_size):
		visible_size = self._last_encoder.get_shape().as_list()[-1]
		self._add_fc_encoder(self._last_encoder, visible_size, hidden_size)
		self._last_encoder = self.encoder_operations[-1]

		encoder = self.encoder_operations[-1]
		def anon(stream_signal):
			self._add_fc_decoder(encoder, stream_signal, visible_size, hidden_size)
		self.build_queue.append(anon)

	def _add_fc_encoder(self, input_to_encode, visible_size, hidden_size):
		print("FC ENC {}".format(hidden_size))
		# Encode is straightforward.  Data always comes in the same way.
		we = tf.Variable(tf.random_normal([visible_size, hidden_size]))
		be = tf.Variable(tf.random_normal([hidden_size,]))
		fc1 = tf.matmul(input_to_encode, we) + be
		act1 = tf.nn.relu(fc1)

		self.encoder_operations.append(act1)
		self.encoder_weights.append(we)
		self.encoder_biases.append(be)

	def _add_fc_decoder(self, signal_from_encoder, input_to_decode, visible_size, hidden_size):
		print("FC DEC {}".format(hidden_size))
		# Decode requires two steps.  First, decoder path.
		wd = tf.Variable(tf.random_normal([hidden_size, visible_size]))
		bd = tf.Variable(tf.random_normal([visible_size, ]))
		fc2 = tf.matmul(input_to_decode, wd) + bd
		act2 = tf.nn.relu(fc2)

		self.decoder_operations.append(act2)
		self.decoder_weights.append(wd)
		self.decoder_biases.append(bd)

		# Second, autoencoder path.
		fc3 = tf.matmul(signal_from_encoder, wd) + bd
		act3 = tf.nn.relu(fc3)
		self.pretrainer_operations.append(act3)

	def add_conv2d(self, filter_height, filter_width, filter_depth, num_filters, strides=None):
		if not strides:
			strides = [1, filter_height, filter_width, 1]
		input_size = self._last_encoder.get_shape().as_list()
		filter_shape = [filter_height, filter_width, filter_depth, num_filters]

		self._add_conv_encoder(self._last_encoder, filter_shape, strides)
		self._last_encoder = self.encoder_operations[-1]

		encoder_ref = self.encoder_operations[-1]
		def anon(signal_to_decode):
			self._add_conv_decoder(encoder_ref, signal_to_decode, input_size, filter_shape, strides)
		self.build_queue.append(anon)

	def _add_conv_encoder(self, input_to_encode, filter_shape, strides):
		print("CONV ENC {}, {}, {}, {}".format(*filter_shape))
		# Encode phase
		we = tf.Variable(tf.random_normal(filter_shape))
		be = tf.Variable(tf.random_normal([filter_shape[-1],]))
		conv = tf.nn.conv2d(input_to_encode, filter=we, strides=strides, padding='SAME') + be
		act1 = tf.nn.relu(conv)
		#pool = tf.nn.max_pool(act1, ksize=[1, filter_shape[1], filter_shape[2], 1], strides=[1, filter_shape[1], filter_shape[2], 1], padding='SAME')
		#norm = tf.nn.lrn(pool, strides[1], bias=1.0, alpha=0.001, beta=0.75)

		self.encoder_operations.append(act1)
		self.encoder_weights.append(we)
		self.encoder_biases.append(be)

	def _add_conv_decoder(self, signal_from_encoder, input_to_decode, input_size, filter_size, strides):
		print("CONV DEC {}, {}".format(input_size, filter_size))
		# Decode phase
		dec_shape = signal_from_encoder.get_shape().as_list()

		# Deconv2D args:
		wd = tf.Variable(tf.random_normal(filter_size))
		bd = tf.Variable(tf.random_normal([input_size[1], input_size[2], input_size[3],]))
		deconv = tf.nn.deconv2d(input_to_decode, filter=wd, strides=strides, padding='SAME', output_shape=input_size) + bd
		act2 = tf.nn.relu(deconv)

		self.decoder_operations.append(act2)
		self.decoder_weights.append(wd)
		self.decoder_biases.append(bd)

		# Autoencode phase
		autoenc = tf.nn.deconv2d(signal_from_encoder, filter=wd, strides=strides, padding='SAME', output_shape=input_size) + bd
		self.pretrainer_operations.append(autoenc)

	def add_pool(self, batch_size, kernel_height, kernel_width, kernel_depth, strides=None):
		input_shape = self._last_encoder.get_shape().as_list()

		if strides is None:
			strides = [1, kernel_height, kernel_width, 1]

		self._add_pool_encoder(self._last_encoder, [batch_size, kernel_height, kernel_width, kernel_depth], strides)
		encoder_reference = self.encoder_operations[-1]
		self._last_encoder = encoder_reference

		def decoder_builder(signal_to_decode):
			self._add_pool_decoder(encoder_reference, signal_to_decode, input_shape, [kernel_height, kernel_width, kernel_depth, input_shape[-1]], strides)
		self.build_queue.append(decoder_builder)

	def _add_pool_encoder(self, input_to_encode, kernel_shape, strides):
		print("POOL ENC {}".format(kernel_shape))
		pool = tf.nn.max_pool(input_to_encode, ksize=kernel_shape, strides=strides, padding='SAME')

		self.encoder_operations.append(pool)
		self.encoder_weights.append(None)
		self.encoder_biases.append(None)

	def _add_pool_decoder(self, signal_from_encoder, input_to_decode, input_shape, kernel_shape, strides):
		print("POOL DEC {} {}".format(input_shape, kernel_shape))
		self._add_conv_decoder(signal_from_encoder, input_to_decode, input_shape, kernel_shape, strides)

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
		last_decoder = self.to_decode
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
input_batch = tf.placeholder(tf.types.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
encoded_batch = tf.placeholder(tf.types.float32, [BATCH_SIZE, REPRESENTATION_SIZE]) # Replace BATCH_SIZE with None
keep_prob = tf.placeholder(tf.types.float32)
autoencoder = ConvolutionalAutoencoder(input_batch, encoded_batch)

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
with tf.Session() as sess:
	# Spin up data iterator.
	generator = gather_batch(sys.argv[1], BATCH_SIZE)

	# Populate autoencoder in session and gather pretrainers.
	autoencoder.add_conv2d(11, 11, IMAGE_DEPTH, 64, strides=[1, 1, 1, 1])
	#autoencoder.add_pool(1, 2, 2, 1, strides=[1, 1, 1, 1])
	autoencoder.add_conv2d(11, 11, 64, 128, strides=[1, 5, 5, 1])
	#autoencoder.add_pool(1, 2, 2, 1, strides=[1, 1, 1, 1])
	autoencoder.add_conv2d(5, 5, 128, 256, strides=[1, 3, 3, 1])
	autoencoder.add_flatten()
	autoencoder.add_fc(128)
	autoencoder.add_fc(32)
	autoencoder.add_fc(REPRESENTATION_SIZE)
	autoencoder.finalize()

	# Collect trainers.
	optimizers = list()
	for layer in range(autoencoder.get_layer_count()-1):
		enc = autoencoder.get_encoder_output(layer)
		dec = autoencoder.get_pretrainer_output(layer)
		l2_cost = tf.reduce_sum(tf.pow(enc - dec, 2))
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(l2_cost)
		optimizers.append(optimizer)

	# Get final ops
	encoder = autoencoder.get_encoder_output()
	decoder = autoencoder.get_decoder_output()

	# Init variables.
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())

	# If we already have a trained network, reload that. The saver doesn't save the graph structure, so we need to build one with identical node names and reload it here.
	# Right now the graph can be loaded with tf.import_graph_def OR the variables can be populated with Restore, but not both (yet).
	# The graph.pbtxt holds graph structure (in model folder).  model-checkpoint has values/weights.
	# TODO: Review when bug is fixed. (2015/11/29)
	if os.path.isfile("./model/checkpoint.model"):
		print("Restored model state.")
		saver.restore(sess, "./model/checkpoint.model")

	# Begin training
	for level, optimizer in enumerate(optimizers):
		for iteration in range(TRAINING_ITERATIONS):
			x_batch = generator.next()
			sess.run(optimizer, feed_dict={input_batch:x_batch})
			if iteration % TRAINING_REPORT_INTERVAL == 0:
				# Checkpoint progress
				print("Finished batch {}".format(iteration))
				saver.save(sess, "./model/checkpoint.model", global_step=iteration)

				# Render output sample
				#encoded, decoded = sess.run([encoder, decoder], feed_dict={input_batch:x_batch, encoded_batch:np.random.uniform(size=(BATCH_SIZE, REPRESENTATION_SIZE))})
				encoded = sess.run(encoder, feed_dict={input_batch:x_batch})

				# Randomly generated sample
				#decoded = sess.run(decoder, feed_dict={encoded_batch:np.random.normal(loc=encoded.mean(), scale=encoded.std(), size=[BATCH_SIZE, REPRESENTATION_SIZE])})
				decoded = sess.run(decoder, feed_dict={encoded_batch:np.random.uniform(low=encoded.min(), high=encoded.max(), size=[BATCH_SIZE, REPRESENTATION_SIZE])})
				#img_tensor = tf.image.encode_jpeg(decoded[0])
				decoded_norm = (decoded[0]-decoded.min())/(decoded.max()-decoded.min())
				img_arr = np.asarray(decoded_norm*255, dtype=np.uint8)
				img = Image.fromarray(img_arr)
				img.save("test_{}_{}.jpg".format(level, iteration))

				# Reconstructed sample ends up looking just like the random sample, so don't waste time making it.
