# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:36:40 2017

@author: Yining Cai
"""

import tensorflow as tf
import numpy as np
from scipy import misc
import random
import pandas as pd
import time

t1 = time.time()
round(time.time() - t1, 2)

n = 50000
images = []
for i in range(1, n+1):
    if i % 1000 == 0:
        print(str(i) + ' out of 50000')
    images.append(misc.imread('C:/github/projects/CIFAR-10/train/' + str(i) + '.png'))
images = np.array(images)
images = images/255
labels = pd.read_csv('C:/github/projects/CIFAR-10/trainLabels.csv')
labels = labels[:n][['label']]
labels = np.array(pd.get_dummies(labels))





x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## conv1
W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
## maxpooling1
h_pool1 = max_pool_2x2(h_conv1)

## conv2
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

## maxpooling2
h_pool2 = max_pool_2x2(h_conv2)

## conv3
W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

## maxpooling3
h_pool3 = max_pool_2x2(h_conv3)

## conv4
W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

## maxpooling4
h_pool4 = max_pool_2x2(h_conv4)



## fully connected
W_fc1 = weight_variable([2*2*128, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool4, [-1, 2*2*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

pred = np.zeros(300000)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  t1 = time.time()
  
  for i in range(15000):
    batch_ind = random.sample(range(50000), 100)
    xbatch = images[batch_ind]
    ybatch = labels[batch_ind]
    if i % 50 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: xbatch, y_: ybatch, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      print('time elapse: ' + str(round(time.time() - t1, 2)) + ' sec')
    train_step.run(feed_dict={x: xbatch, y_: ybatch, keep_prob: 0.5})
    
  
  for j in range(1, 300001):
    if j % 1000 == 0:
      print(str(j) + ' out of 300000')
      print('time elapse: ' + str(round(time.time() - t1, 2)) + ' sec')
    images_test = misc.imread('C:/github/projects/CIFAR-10/test/' + str(j) + '.png')
    images_test = np.array(images_test)
    images_test = images_test/255
    pred_softmax = sess.run(y_conv, {x: [images_test], keep_prob: 1.0})
    pred_class = np.array(pred_softmax)
    pred[j-1] = int(np.argmax(pred_class, 1))
    
    
pred1 = pd.DataFrame({'id': range(1, 300001), 'label_id': pred})
lookup = pd.DataFrame({'label_id': range(10), 'label': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']})
pred2 = pred1.merge(lookup)
pred2 = pred2[['id', 'label']]
pred2 = pred2.sort_values(['id'])
pred2.to_csv('C:/github/projects/CIFAR-10/pred.csv', index = False)




