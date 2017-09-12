from vgg16 import vgg16
import numpy as np
import tensorflow as tf

weight_file = '/home/dashmoment/dataset/vgg16_weights.npz'
weights = np.load(weight_file)
keys = sorted(weights.keys())

sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vgg16.vgg16(imgs, weight_file , sess)

a = vgg.parameters