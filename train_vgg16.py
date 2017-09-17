import numpy as np
import tensorflow as tf
import os
import pandas as pd
import sys
import random

sys.path.append("utility")
sys.path.append("vgg16")
import model_zoo as mz
from imagenet_classes import class_names
import data_utility as du
import log_utility as log_u
import config


conf = config.config("home")
data_u = du.data_utility("home")

with tf.name_scope('preprocess') as scope:
    mean = tf.constant([114.156, 121.907, 126.488], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    images = conf.imgs-mean
    
with tf.name_scope("vgg16") :   
    mz = mz.model_zoo(images, conf.is_training, conf.dropout, "vgg16_aic_v1")
    logits = mz.build_model()
    prediction = tf.arg_max(tf.nn.softmax(logits),1)

with tf.name_scope("training"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = conf.label, logits=logits, name="loss"))
    train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)

with tf.name_scope("accuracy"):
	true_positive = tf.reduce_sum(tf.cast(tf.equal(prediction, conf.label), tf.int64))

log_collection = {"loss":loss, "accuracy":true_positive}

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	log = log_u.logger(sess, checkpoint_path=conf.checkpoint_path, summary_path=conf.summary_path)
	restore_pretrain = log_u.resotre_pretrained_weight(sess,conf.pretrained_weight_path)
	train_log = log.build_logger("train",log_collection)
	test_log = log.build_logger("test",log_collection)

	restore = log.restore_model()
	if restore != True:
		restore_pretrain.vgg16_imagenet()


	for epoch in range(conf.Nepoch):

		random.shuffle(conf.train_idx_list)

		for step in range(len(conf.train_idx_list)//conf.batch_size):
			
			print("Epoch:{}, Iteration:{}".format(epoch, step))

			train_batch = data_u.get_batch('train', conf.train_idx_list,step)

			feed_dict = {conf.imgs:train_batch[0],conf.label:train_batch[1], conf.learning_rate:1e-4, conf.is_training:True, conf.dropout:0.5}

			sess.run(train_op, feed_dict = feed_dict)

			iterations = step + len(conf.train_idx_list)*epoch

			if step % conf.train_log_step == 0:
				loss_out, acc,train_sum = sess.run([loss, true_positive, train_log], feed_dict = feed_dict)
				log.add_summary(train_sum,iterations)
				log.save_model(iterations)

			if step % conf.test_log_step == 0:

				random.shuffle(conf.validation_idx_list)

				test_batch = data_u.get_batch('validation', conf.validation_idx_list,0)
				feed_dict = {conf.imgs:test_batch[0],conf.label:test_batch[1], conf.learning_rate:1e-4, conf.is_training:False, conf.dropout:1}
				loss_out, acc,test_sum= sess.run([loss,true_positive ,test_log], feed_dict = feed_dict)
				log.add_summary(test_sum,iterations)
				
				
