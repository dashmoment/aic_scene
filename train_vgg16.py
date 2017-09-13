import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

import sys
sys.path.append("utility")
sys.path.append("vgg16")
import model_zoo as mz
from imagenet_classes import class_names



weight_file = '/home/dashmoment/dataset/vgg16_weights.npz'
weights = np.load(weight_file)
keys = sorted(weights.keys())

imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
is_training = tf.placeholder(tf.bool, name='is_training')
dropout = tf.placeholder(tf.float32, name='dropout')


with tf.name_scope('preprocess') as scope:
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    images = imgs-mean
mz = mz.model_zoo(images, is_training, dropout, "vgg16_aic_v1")
logits = mz.build_model()


sess = tf.Session()
sess.run(tf.global_variables_initializer())


variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)

variabel_list = []
for k, v in zip(variables_names, values):
    print ("Variable: ", k)
    variabel_list.append(k)
#    print ("Shape: ", v.shape)
#    print (v)
#w0 = sess.run(tf.get_default_graph().get_tensor_by_name(variabel_list[0]))
for idx in range(32):
    
    tmp = tf.get_default_graph().get_tensor_by_name(variabel_list[idx])
    sess.run(tf.assign(tmp,weights[keys[idx]]))


img1 = imread('vgg16/laska.png', mode='RGB')
img1 = imresize(img1, (224, 224))

prob = sess.run(tf.nn.softmax(logits), feed_dict={imgs: [img1], is_training:False, dropout:1})[0]
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print ("class:{}, Prob:{}".format(class_names[p], prob[p]))
    
    
'''
weasel 0.693386
polecat, fitch, foulmart, foumart, Mustela putorius 0.175388
mink 0.122086
black-footed ferret, ferret, Mustela nigripes 0.00887066
otter 0.000121083
'''