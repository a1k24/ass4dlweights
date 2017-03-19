'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:Akash Mandal
Roll No.:13CS10006

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

batch_size = 100
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

sess = tf.InteractiveSession()
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def train(trainX, trainY):
	'''
	Complete this function.
	'''
	sess.run(tf.global_variables_initializer())
	trainX = trainX.reshape(-1,784)
	b = np.zeros((len(trainY), 10))
	b[np.arange(len(trainY)),trainY] = 1
	trainY = b
	for iter in range(12):
		print "epoch", iter
		print int(trainX.shape[0] / batch_size)
		for i in range(int(trainX.shape[0] / batch_size)):
			n = min(batch_size, trainX.shape[0]-i*batch_size)
			train_step.run(feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n], keep_prob: 0.5})
			loss = sess.run(cross_entropy, feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n], keep_prob: 0.5})
			print loss
			if  i == int(trainX.shape[0] / batch_size)-1:
				train_accuracy = accuracy.eval(feed_dict={
					x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n], keep_prob: 1.0})
				print("step %d, training accuracy %g"%(iter, train_accuracy))
	saver.save(sess,"model_cnn")





def test(testX):
	'''
	Complete this function.
	This function must read the weight files and
	return the predicted labels.
	The returned object must be a 1-dimensional numpy array of
	length equal to the number of examples. The i-th element
	of the array should contain the label of the i-th test
	example.
	'''
	new_saver = tf.train.import_meta_graph('model_cnn.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	testX = testX.reshape(-1,784)
	y1 = sess.run(y_conv,feed_dict={x: testX, keep_prob: 1.0})

	return np.argmax(y1,1)
