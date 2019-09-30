import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cnn_utils import *
import os

def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,minibatch_size,print_cost,isPlot):
	seed = 3
	(m, n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	X,Y,keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)
	Z6 = forward_propagation(X,keep_prob)
	cost = compute_loss(Z6, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m / minibatch_size) 
			minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed) 
			for minibatch in minibatches:
				(minibatch_X,minibatch_Y) = minibatch
				_ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y,keep_prob:0.5})

				minibatch_cost += temp_cost / num_minibatches

			if print_cost:
				if epoch % 5 == 0:
					print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

			if epoch % 1 == 0:
				costs.append(minibatch_cost)
		if isPlot:
			plt.plot(np.squeeze(costs))
			plt.ylabel('cost')
			plt.xlabel('iterations (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()

		saver.save(sess,"model/save_net.ckpt")

		corrent_prediction = tf.equal(tf.argmax(Z6,1) , tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train,keep_prob:1.0})
		test_accuary = accuracy.eval({X: X_test, Y: Y_test,keep_prob:1.0})
		print("训练集准确度：" + str(train_accuracy))
		print("测试集准确度：" + str(test_accuary))

if __name__== "__main__":
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig = load_data()
	X_train,Y_train,X_test,Y_test = init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig)
	print(X_train.shape)
	print(Y_train.shape)
	print(X_test.shape)
	print(Y_test.shape)
	model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 20,minibatch_size=64,print_cost=True,isPlot=True)

