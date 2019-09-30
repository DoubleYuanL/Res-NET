import math
import numpy as np
import h5py
import tensorflow as tf

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_data():
    X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = load_dataset()
    return X_train_orig , Y_train_orig , X_test_orig , Y_test_orig 

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig ):
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    return X_train, Y_train, X_test, Y_test

def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
    Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
    dropout = tf.placeholder(tf.float32)
    return X,Y, dropout

def conv2d_block(X,IN,OUT):
    #主路径
    Z1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=IN, kernel_size=[1,1], stride=[2,2], padding="VALID", activation_fn=None)
    Z1 = tf.layers.batch_normalization(Z1,3, training=True)
    Z1 = tf.nn.relu(Z1)

    Z2 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=IN, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=None)
    Z2 = tf.layers.batch_normalization(Z2,3, training=True)
    Z2 = tf.nn.relu(Z2)

    Z3 = tf.contrib.layers.conv2d(inputs=Z2, num_outputs=OUT, kernel_size=[1,1], stride=[1,1], padding="VALID", activation_fn=None)
    Z3 = tf.layers.batch_normalization(Z3,3, training=True)

    #捷径
    Z4 = tf.contrib.layers.conv2d(inputs=X, num_outputs=OUT, kernel_size=[1,1], stride=[2,2], padding="VALID", activation_fn=None)
    Z4 = tf.layers.batch_normalization(Z4,3, training=True)

    #相加 激活
    ZX = tf.add(Z3,Z4)
    AX = tf.nn.relu(ZX)
    return AX

def identity_block(X,IN,OUT):
    #主路径
    Z1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=IN, kernel_size=[1,1], stride=[1,1], padding="VALID", activation_fn=None)
    B1 = tf.layers.batch_normalization(Z1,3, training=True)
    A1 = tf.nn.relu(B1)

    Z2 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=IN, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=None)
    B2 = tf.layers.batch_normalization(Z2,3, training=True)
    A2 = tf.nn.relu(B2)

    Z3 = tf.contrib.layers.conv2d(inputs=Z2, num_outputs=OUT, kernel_size=[1,1], stride=[1,1], padding="VALID", activation_fn=None)
    B3 = tf.layers.batch_normalization(Z3,3, training=True)

    #捷径
    ZS = X
    #相加 激活
    ZX = tf.add(Z3,ZS)
    AX = tf.nn.relu(ZX)
    return AX

def forward_propagation(X,keep_prob):
    h = 64
    j = 128
    k = 256
    l = 512
    m = 1024
    Z1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=[7,7], stride=[2,2], padding="SAME",activation_fn=None)
    Z1 = tf.layers.batch_normalization(Z1,3, training=True)
    Z1 = tf.nn.relu(Z1)
    Z1 = tf.contrib.layers.max_pool2d(inputs=Z1, kernel_size=[3,3], stride=[2,2], padding='VALID')

    RS1 = conv2d_block(Z1,h,j)
    RS1 = identity_block(RS1,h,j)
    RS1 = identity_block(RS1,h,j)
    print(RS1.shape)

    RS2 = conv2d_block(RS1,j,k)
    RS2 = identity_block(RS2,j,k)
    RS2 = identity_block(RS2,j,k)
    RS2 = identity_block(RS2,j,k)
    print(RS2.shape)

    # RS3 = conv2d_block(RS2,k,l)
    # RS3 = identity_block(RS3,k,l)
    # RS3 = identity_block(RS3,k,l)
    # RS3 = identity_block(RS3,k,l)
    # RS3 = identity_block(RS3,k,l)
    # RS3 = identity_block(RS3,k,l)
    # print(RS3.shape)

    # RS4 = conv2d_block(RS3,l,m)
    # RS4 = identity_block(RS4,l,m)
    # RS4 = identity_block(RS4,l,m)
    # RS4 = identity_block(RS4,l,m)
    # print(RS4.shape)

    P1 = tf.contrib.layers.avg_pool2d(inputs=RS2, kernel_size=[4,4], stride=[1,1], padding='SAME')

    Fa1 = tf.contrib.layers.flatten(P1)
    F1 = tf.contrib.layers.fully_connected(Fa1,1024,activation_fn=tf.nn.relu)
    F1 = tf.nn.dropout(F1, keep_prob)
    F2 = tf.contrib.layers.fully_connected(F1,512,activation_fn=tf.nn.relu)
    D2 = tf.nn.dropout(F2, keep_prob)
    ZR = tf.contrib.layers.fully_connected(F2,6,activation_fn=None)
    return ZR

def compute_loss(Z,Y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z, labels = Y))
    return loss 