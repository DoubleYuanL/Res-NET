import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from cnn_utils import *
import cv2

def predict():
    X,_,keep_prob = create_placeholder(64, 64, 3, 6)
    output = forward_propagation(X,keep_prob)
    result = tf.argmax(output,1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint("model/"))
        num = 1
        while 1:
            my_image = "datasets/sample/" + str(num) + ".jpg"    
            num_px = 64
            fname =  my_image 
            image = np.array(ndimage.imread(fname, flatten=False))
            my_predicted_image = image.reshape((1,64,64,3))/255

            my_predicted_image,pre = sess.run([result,output], feed_dict={X:my_predicted_image,keep_prob:1.0})
            print(pre)
            plt.imshow(image) 
            print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
            plt.show()
            num = num + 1

if __name__ == '__main__':
    predict()
