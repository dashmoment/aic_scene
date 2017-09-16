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


### Weight file shoual be .npz file
def load_pretrain_weight(sess, wieght_file):
    
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    
    variables_names = [v.name for v in tf.trainable_variables()]
    
    graph = tf.get_default_graph()
    
    for i in range(len(keys)):
        
        #print(variables_names[i], keys[i])
        var = graph.get_tensor_by_name(variables_names[i]) 
        sess.run(tf.assign(var, weights[keys[i]]))

class logger:

	def __init__(self, sess, checkpoint_path = None, checkpoint_name = 'model.ckpt',summary_path = None):

		self.checkpoint_path = checkpoint_path
		self.summary_path = summary_path
		self.sess = sess
		self.checkpoint_file = None
		self.summary_writer = None

		if checkpoint_path != None: 
			if not os.path.exists(checkpoint_path):
				os.makedirs(checkpoint_path)
			self.checkpoint_file = os.path.join(checkpoint_path,checkpoint_name) 
			self.saver = tf.train.Saver()

		if summary_path != None:
			if not os.path.exists(summary_path):
				os.makedirs(summary_path)
			self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)    

	def build_logger(self, collection, log_content, summary_type_fn = tf.summary.scalar):

		for key in log_content:

			summary_type_fn(key, log_content[key], collections=[collection])
		merged_summary = tf.summary.merge_all(collection) 

		return merged_summary

	def add_summary(self, summary, step):
		self.summary_writer.add_summary(summary, step)

	def save_model(self,step):
		self.saver.save(self.sess,
                        os.path.join(self.checkpoint_path, self.checkpoint_name),
                        global_step=step)

	def restore_model(self):
		
		if tf.train.get_checkpoint_state(self.checkpoint_path):
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
			return True
		else:
			print("Fail to resotre model")
			return False



with tf.name_scope("plcaholders"):
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
    label = tf.placeholder(tf.int64, [None], name="labels")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')


with tf.name_scope('preprocess') as scope:
    mean = tf.constant([114.156, 121.907, 126.488], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    images = imgs-mean
    
    
with tf.name_scope("vgg16") :   
    mz = mz.model_zoo(images, is_training, dropout, "vgg16_aic_v1")
    logits = mz.build_model()
    prediction = tf.arg_max(tf.nn.softmax(logits),1)

with tf.name_scope("training"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits=logits, name="loss"))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope("accuracy"):
	true_positive = tf.reduce_sum(tf.cast(tf.equal(prediction, label), tf.int64))


data_root = '/home/dashmoment/dataset/ai_challenger_scene'
#data_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ai_challenger_scene'
validation_set = os.path.join(data_root, 'validation')
training_set = os.path.join(data_root, 'train')
ann_val_path = os.path.join(validation_set, 'scene_validation_annotations_20170908.json')
ann_train_path = os.path.join(training_set, 'scene_train_annotations_20170904.json')
ann_val = pd.read_json(ann_val_path)
ann_train = pd.read_json(ann_train_path)
train_list = list(range(len(ann_train)))
validation_list = list(range(len(ann_val)))

Nepoch = 100000
batch_size = 16

log_collection = {"loss":loss, "accuracy":true_positive}
checkpoint_path = 'model'
summary_path = 'log'

with tf.Session() as sess:

	log = logger(sess, checkpoint_path=checkpoint_path, summary_path=summary_path)
	train_log = log.build_logger("train",log_collection)
	test_log = log.build_logger("test",log_collection)

	sess.run(tf.global_variables_initializer())

	restore = log.restore_model()
	if restore != True:
		weight_file = '/home/dashmoment/dataset/vgg16_weights.npz'
		load_pretrain_weight(sess, weight_file)


	for epoch in range(Nepoch):

		random.shuffle(train_list)

		for step in range(len(train_list)//batch_size):
			
			print("Epoch:{}, Iteration:{}".format(epoch, step))

			train_batch = du.get_batch(training_set, ann_train, batch_size, train_list, step)


			feed_dict = {imgs:train_batch[0],label:train_batch[1], learning_rate:1e-4, is_training:True, dropout:0.5}

			sess.run(train_op, feed_dict = feed_dict)

			if step % 100 == 0:
				loss_out, acc,train_sum = sess.run([loss, true_positive, train_log], feed_dict = feed_dict)
				log.add_summary(train_sum,step + len(train_list)*epoch)

			if step % 500 == 0:

				random.shuffle(validation_list)
				test_batch = du.get_batch(validation_set, ann_val, batch_size, validation_list, step)
				feed_dict = {imgs:test_batch[0],label:test_batch[1], learning_rate:1e-4, is_training:False, dropout:1}
				loss_out, acc,test_sum= sess.run([loss,true_positive ,test_log], feed_dict = feed_dict)
				log.add_summary(test_sum,step + len(train_list)*epoch)
				
				
