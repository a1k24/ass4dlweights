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

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.random_normal([784, 200]))
b = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.random_normal([200, 10]))
b2 = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)
H = tf.nn.relu(tf.matmul(x,W)+b)
y = tf.matmul(H,W2)+b2
y_ = tf.placeholder(tf.float32, [None, 10])
batch_size = 100
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
saver = tf.train.Saver()
sess = tf.InteractiveSession()

def train(trainX, trainY):
	'''
	Complete this function.
	'''
	tf.global_variables_initializer().run()
	trainX = trainX.reshape(-1,784)
	b = np.zeros((len(trainY), 10))
	b[np.arange(len(trainY)),trainY] = 1
	trainY = b
	for iter in range(10):
		print "epoch", iter
		print int(trainX.shape[0] / batch_size)
		batch_total = 0
		for i in range(int(trainX.shape[0] / batch_size)):
		# for i in range(1):
			# print trainX[0][1], trainY[0]
			n = min(batch_size, trainX.shape[0]-i*batch_size)
			sess.run(train_step, feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n]})
			loss = sess.run(cross_entropy, feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n]})
			# w = sess.run(W, feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n]})
			# print loss
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			train_accuracy = sess.run(accuracy, feed_dict={x: trainX[i*batch_size:i*batch_size+n], y_: trainY[i*batch_size:i*batch_size+n]})
			batch_total += train_accuracy
		print batch_total/int(trainX.shape[0] / batch_size)
	saver.save(sess,"model")

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
	new_saver = tf.train.import_meta_graph('model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	testX = testX.reshape(-1,784)
	y1 = sess.run(y,feed_dict={x: testX})
	return np.argmax(y1,1)
