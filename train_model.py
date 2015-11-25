#!/usr/bin/env python
import sys, os
from glob import iglob

from PIL import Image
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 10000
TRAINING_DROPOUT_RATE = 0.8
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 64
BATCH_SIZE = 5
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

# Create model

class ConvolutionalAutoencoder(object):
	def __init__(self, to_encode, to_decode):
		self.to_encode = to_encode
		self.to_decode = to_decode

		self.encoder_inputs = list()
		self.decoder_inputs = list()
		self.encoder_operations = list()
		self.decoder_operations = list()
		self.pretrainer_operations = list() # Used for layer-wise pretraining. pretrainer_operations[0] is encoder -> hidden state -> decoder. 
		self.encoder_weights = list()
		self.encoder_biases = list()
		self.decoder_weights = list()
		self.decoder_biases = list()

	def add_fc_layer(self, to_encode, to_decode, input_size, output_size):
		# Add encode/decode
		self.encoder_inputs.append(to_encode)
		self.decoder_inputs.append(to_decode)

		# Encode is straightforward.  Data always comes in the same way.
		we = tf.Variable(tf.random_normal([input_size, output_size]))
		be = tf.Variable(tf.random_normal([output_size,]))
		fc1 = tf.matmul(to_encode, we) + be
		act1 = tf.nn.relu(fc1)

		self.encoder_operations.append(act1)
		self.encoder_weights.append(we)
		self.encoder_biases.append(wb)

		# Decode requires two steps.  First, decoder path.
		wd = tf.Variable(tf.random_normal([output_size, input_size]))
		bd = tf.Variable(tf.random_normal([input_size, ]))
		fc2 = tf.matmul(to_decode, wd) + bd
		act2 = tf.nn.relu(fc2)

		self.decoder_operations.append(act2)
		self.decoder_weights.append(wd)
		self.decoder_biases.append(bd)

		# Second, autoencoder path.
		fc3 = tf.matmul(act1, wd) + bd
		act3 = tf.nn.relu(fc3)
		self.pretrainer_operations.append(act3)

	def add_conv2d(self, batch_size, input_height, input_width, input_depth, num_filters):
		# Encode phase
		to_encode = tf.placeholder(tf.types.float32, [None, input_height, input_width, input_depth])

		we = tf.Variable(tf.random_normal([input_height, input_width, input_depth, num_filters]))
		be = tf.Variable(tf.random_normal([num_filters,]))
		conv = tf.nn.conv2d(to_encode, filter=we, strides=[1, 1, 1, 1], padding='SAME') + be
		act1 = tf.nn.relu(conv)
		pool = tf.nn.max_pool(act1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')
		norm = tf.nn.lrn(pool, 5, bias=1.0, alpha=0.001, beta=0.75)

		self.encoder_inputs.append(to_encode)
		self.encoder_operations.append(norm)
		self.encoder_weights.append(we)
		self.encoder_biases.append(be)

		# Decode phase
		dec_shape = norm.get_shape().as_list()
		to_decode = tf.placeholder(tf.types.float32, [None, dec_shape[0], dec_shape[1], dec_shape[2]])

		wd = tf.Variable(tf.random_normal([input_height, input_width, input_depth, num_filters]))
		bd = tf.Variable(tf.random_normal([num_filters,]))
		deconv = tf.nn.deconv2d(to_decode, filter=wd, strides=[1, 1, 1, 1], padding='SAME', output_shape=[batch_size, input_height, input_width, input_depth]) + bd
		act2 = tff.nn.relu(deconv)

		self.decoder_inputs.append(to_decode)
		self.decoder_operations.append(act2)
		self.decoder_weights.append(wd)
		self.decoder_biases.append(bd)

		# Autoencode phase
		autoenc = tf.nn.deconv2d(norm, filter=wd, strides=[1, 1, 1, 1], padding='SAME', output_shape=[batch_size, input_height, input_width, input_depth]) + bd
		self.pretrainer_operations.append(autoenc)

	def add_flatten(self, to_encode, to_decode, batch_size, input_height, input_width, input_depth):
		# Encode
		flatten = tf.reshape(to_encode, [batch_size, input_height*input_width*input_depth])

		self.encoder_operations.append(flatten)
		self.encoder_weights.append(None)
		self.encoder_biases.append(None)

		# Decode
		unflatten = tf.reshape(to_decode, [batch_size, input_height, input_width, input_depth])

		self.decoder_operations.append(unflatten)
		self.decoder_weights.append(None)
		self.decoder_biases.append(None)

		# Not strictly necessary, but...
		autoenc = tf.reshape(flatten, [batch_size, input_height, input_width, input_depth])
		self.pretrainer_operations.append(autoenc)

	def get_output_shape(self):
		return self.encoder_operations[-1].get_shape()

	def get_layer_trainer(self, layer_num):
		operation_chain = list()
		operation_chain.append(self.to_encode)
		for i in range(layer_num):
			last_result = tf.assign(last_result, self.encoder_operations[i])


# Define objects
keep_prob = tf.placeholder(tf.types.float32) #dropout
batch_shape = tf.placeholder(tf.types.int32, shape=(4,))
representation_size = tf.placeholder(tf.types.int32)
input_batch = tf.placeholder(tf.types.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
encoded_batch = tf.placeholder(tf.types.float32, [None, REPRESENTATION_SIZE])
keep_prob = tf.placeholder(tf.types.float32)

autoencoder = ConvolutionalAutoencoder(input_batch, encoded_batch)
out_shape = [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
autoencoder.add_conv2d(1, out_shape[1], out_shape[2], out_shape[3], 128)
out_shape = autoencoder.get_output_shape().as_list()
autoencoder.add_conv2d(1, out_shape[1], out_shape[2], out_shape[3], 128)
out_shape = autoencoder.get_output_shape().as_list()

# Define goals
l1_cost = tf.reduce_mean(tf.abs(input_batch - autoencoder))
l2_cost = tf.reduce_sum(tf.pow(input_batch - autoencoder, 2))
cost = l2_cost
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Define data-source iterator
def gather_batch(file_glob, batch_size):
	reader = tf.WholeFileReader()
	while True:
		image_batch = list()
		batch = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=np.float)
		for index, filename in zip(range(batch_size), iglob(file_glob)):
			img = Image.open(filename)
			batch[index,:,:,:] = np.asarray(img)/255.0
		yield batch
			
# Run!
with tf.Session() as sess:
	generator = gather_batch(sys.argv[1], BATCH_SIZE)
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())
	for iteration in range(TRAINING_ITERATIONS):
		x_batch = generator.next()
		sess.run(optimizer, feed_dict={input_batch:x_batch, keep_prob: TRAINING_DROPOUT_RATE, batch_shape: tf.constant([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])})
		if iteration % TRAINING_REPORT_INTERVAL == 0:
			l1_score, l2_score = sess.run([l1_cost, l2_cost], feed_dict={input_batch:x_batch, keep_prob:1.0})
			print("Iteration {}: L1 {}  L2 {}".format(iteration, l1_score, l2_score))
			saver.save(sess, "checkpoint.model", global_step=iteration)
			#fout = open("example.jpg", 'wb')
			#tf.image.encode_jpg(

