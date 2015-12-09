#!/usr/bin/env python
import sys, os
from glob import glob
from random import choice
from io import BytesIO

from PIL import Image
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 100000
TRAINING_DROPOUT_RATE = 0.8
TRAINING_REPORT_INTERVAL = 100
REPRESENTATION_SIZE = 128
BATCH_SIZE = 1
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 3

# Create model
def build_encoder(stream_to_encode, stream_to_decode):
	"""Given the two streams, returns an encoder output and a decoder output."""
	w0 = tf.Variable(tf.random_normal([5, 5, IMAGE_DEPTH, 128]))
	b0 = tf.Variable(tf.random_normal([128,]))
	conv0 = tf.nn.conv2d(stream_to_encode, filter=w0, strides=[1, 5, 5, 1], padding='SAME') + b0
	act0 = tf.nn.relu(conv0)
	# Max pooling along depth not yet supported.  :(
	#pool0 = tf.nn.max_pool(act0, ksize=[1, 1, 1, 128], strides=[1, 1, 1, 128], padding='SAME') # Squash depth, 1x1x128 -> 1x1x1

	w1 = tf.Variable(tf.random_normal([5, 5, 128, 64]))
	b1 = tf.Variable(tf.random_normal([64,]))
	conv1 = tf.nn.conv2d(act0, filter=w1, strides=[1, 1, 1, 1], padding='SAME') + b1
	act1 = tf.nn.relu(conv1)
	#pool1 = tf.nn.max_pool(act1, ksize=[1, 1, 1, 64], strides=[1, 1, 1, 64], padding='SAME') # Squash horizontally, leaving 1x1x64 per 5x5x128 chunk.

	flat = tf.reshape(act1, [BATCH_SIZE, -1])
	
	w3 = tf.Variable(tf.random_normal([flat.get_shape().as_list()[-1], 512]))
	b3 = tf.Variable(tf.random_normal([512,]))
	mmul3 = tf.matmul(flat, w3) + b3
	act3 = tf.nn.relu(mmul3)

	w4 = tf.Variable(tf.random_normal([512, REPRESENTATION_SIZE]))
	b4 = tf.Variable(tf.random_normal([REPRESENTATION_SIZE,]))
	mmul4 = tf.matmul(act3, w4) + b4
	act4 = tf.nn.relu(mmul4)

	encoder = tf.identity(act4, name='encoder_output')

	w5 = tf.Variable(tf.random_normal([REPRESENTATION_SIZE, 512]))
	b5 = tf.Variable(tf.random_normal([512,]))
	mmul5_dec = tf.matmul(stream_to_decode, w5) + b5
	act5_dec = tf.nn.relu(mmul5_dec)
	mmul5_ae = tf.matmul(encoder, w5) + b5
	act5_ae = tf.nn.relu(mmul5_ae)

	w6 = tf.Variable(tf.random_normal([512, flat.get_shape().as_list()[-1]]))
	b6 = tf.Variable(tf.random_normal([flat.get_shape().as_list()[-1],]))
	mmul6_dec = tf.matmul(act5_dec, w6) + b6
	act6_dec = tf.nn.relu(mmul6_dec)
	mmul6_ae = tf.matmul(act5_ae, w6) + b6
	act6_ae = tf.matmul(act5_ae, w6)

	unflat_dec = tf.reshape(act6_dec, act1.get_shape().as_list())
	unflat_ae = tf.reshape(act6_ae, act1.get_shape().as_list())

	w7 = tf.Variable(tf.random_normal([5, 5, 128, 64]))
	b7 = tf.Variable(tf.random_normal(act0.get_shape().as_list()[1:]))
	deconv8_dec = tf.nn.deconv2d(unflat_dec, filter=w7, strides=[1, 1, 1, 1], padding='SAME', output_shape=act0.get_shape().as_list()) + b7
	act8_dec = tf.nn.relu(deconv8_dec)
	deconv8_ae = tf.nn.deconv2d(unflat_ae, filter=w7, strides=[1, 1, 1, 1], padding='SAME', output_shape=act0.get_shape().as_list()) + b7
	act8_ae = tf.nn.relu(deconv8_ae)

	w8 = tf.Variable(tf.random_normal([5, 5, IMAGE_DEPTH, 128]))
	deconv9_dec = tf.nn.deconv2d(act8_dec, filter=w8, strides=[1, 5, 5, 1], padding='SAME', output_shape=stream_to_encode.get_shape().as_list())
	deconv9_ae = tf.nn.deconv2d(act8_ae, filter=w8, strides=[1, 5, 5, 1], padding='SAME', output_shape=stream_to_encode.get_shape().as_list())

	return encoder, deconv9_dec, deconv9_ae

# Define objects
input_batch = tf.placeholder(tf.types.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
encoded_batch = tf.placeholder(tf.types.float32, [BATCH_SIZE, REPRESENTATION_SIZE]) # Replace BATCH_SIZE with None
keep_prob = tf.placeholder(tf.types.float32)

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

	# Get final ops
	encoder, decoder, autoenc = build_encoder(input_batch, encoded_batch)
	l2_cost = tf.reduce_sum(tf.pow(input_batch - autoenc, 2))
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(l2_cost)

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
			img.save("test_{}.jpg".format(iteration))

			# Reconstructed sample ends up looking just like the random sample, so don't waste time making it.
