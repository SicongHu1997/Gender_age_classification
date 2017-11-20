from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import csv
from PIL import Image
import tensorflow as tf
import numpy as np

with open('txt/fold_0_data.txt') as f:
    i_train=[]
    gender_train=[]
    count=0
    f_tsv = csv.reader(f, delimiter='\t')
    for row in f_tsv:
        if count>150:
            break
        if count>0:
            img=Image.open("aligned/"+row[0]+"/landmark_aligned_face."+row[2]+"."+row[1])
            img.thumbnail((272,272), resample=Image.LANCZOS)
            arr = np.array(img)/256
            i_train.append(arr)
            if row[4]=="m":
                gender_train.append([1.0,0.0])
            else:
                gender_train.append([0.0,1.0])
        count+=1

with open('txt/fold_1_data.txt') as f:
    i_test=[]
    gender_test=[]
    count=0
    f_tsv = csv.reader(f, delimiter='\t')
    for row in f_tsv:
        if count>50:
            break
        if count>0:
            img=Image.open("aligned/"+row[0]+"/landmark_aligned_face."+row[2]+"."+row[1])
            img.thumbnail((272,272), resample=Image.LANCZOS)
            arr = np.array(img)/256
            i_test.append(arr)
            if row[4]=="m":
                gender_test.append([1.0,0.0])
            else:
                gender_test.append([0.0,1.0])
        count+=1

def deepnn(x_image):
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([68 * 68 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 68*68*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



x = tf.placeholder(tf.float32, [None, 272, 272, 3])
y_ = tf.placeholder(tf.float32, [None, 2])
y_conv, keep_prob = deepnn(x)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        c = i%10
        if i % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: i_test, y_: gender_test, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: i_train[c*15:c*15+15], y_: gender_train[c*15:c*15+15], keep_prob: 0.5})

