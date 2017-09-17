import tensorflow as tf
import os
import numpy as np

class resotre_pretrained_weight:

	def __init__(self, sess,weight_file):

		self.weight_file = weight_file
		self.sess = sess
	def vgg16_imagenet(self):

		weights = np.load(self.weight_file)
		keys = sorted(weights.keys())
		variables_names = [v.name for v in tf.trainable_variables()]
		graph = tf.get_default_graph()

		for i in range(len(keys)):
			print("Var {} restored by {}".format(variables_names[i], keys[i]))
			var = graph.get_tensor_by_name(variables_names[i]) 
			self.sess.run(tf.assign(var, weights[keys[i]]))


class logger:

	def __init__(self, sess, checkpoint_path = None, checkpoint_name = 'model.ckpt',summary_path = None):

		self.checkpoint_path = checkpoint_path
		self.checkpoint_name = checkpoint_name
		self.summary_path = summary_path
		self.sess = sess
		self.checkpoint_file = None
		self.summary_writer = None
		

		if checkpoint_path != None: 
			if not os.path.exists(checkpoint_path):
				os.makedirs(checkpoint_path)
			self.checkpoint_file = os.path.join(self.checkpoint_path,self.checkpoint_name) 
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
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_path))
			return True
		else:
			print("Fail to resotre model")
			return False