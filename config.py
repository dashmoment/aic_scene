import os
import pandas as pd
import tensorflow as tf

class config:

	def __init__(self, config):

		self.batch_size = 1
		self.test_batch_size = 1
		self.Nepoch = 100000
		self.train_log_step = 200
		self.test_log_step = 500
		self.config = config
		self.img_size = 224

		self.get_config()
		

		self.validation_set = os.path.join(self.data_root, 'validation')
		self.training_set = os.path.join(self.data_root, 'train')
		self.test_set = None
		self.ann_val = pd.read_json(os.path.join(self.validation_set, 'scene_validation_annotations_20170908.json'))
		self.ann_train = pd.read_json(os.path.join(self.training_set, 'scene_train_annotations_20170904.json'))
		self.ann_test = None
		self.validation_idx_list = list(range(len(self.ann_val)))
		self.train_idx_list = list(range(len(self.ann_train)))
		self.test_idx_list = None

		with tf.name_scope("plcaholders"):
		    self.imgs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3], name="images")
		    self.label = tf.placeholder(tf.int64, [None], name="labels")
		    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		    self.is_training = tf.placeholder(tf.bool, name='is_training')
		    self.dropout = tf.placeholder(tf.float32, name='dropout')
		

	def get_config(self):

		try:
			conf = getattr(self, self.config)
			conf()

		except:
			print("Can not find configuration")
			return

	def home(self):

		self.batch_size = 16
		self.data_root = '/home/dashmoment/dataset/ai_challenger_scene'
		self.checkpoint_path = '../aic_log/model'
		self.meta_file = os.path.join(self.checkpoint_path, 'model.ckpt-0.meta')
		self.summary_path = '../aic_log/log'
		self.pretrained_weight_path = '/home/dashmoment/dataset/vgg16_weights.npz'
		self.pretrained_resnet_path = '/home/dashmoment/dataset/pretrained/resnet-pretrained'
		self.pretrained_resnet50_meta = os.path.join(self.pretrained_resnet_path, 'ResNet-L50.meta')
		self.pretrained_resnet152_meta = os.path.join(self.pretrained_resnet_path, 'ResNet-L152.meta')
		self.pretrained_resnet50_ckpt = os.path.join(self.pretrained_resnet_path, 'ResNet-L50.ckpt')
		self.pretrained_resnet152_ckpt = os.path.join(self.pretrained_resnet_path, 'ResNet-L152.ckpt')


	def office(self):
		self.batch_size = 32
		self.data_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/ai_challenger_scene'
		self.checkpoint_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/model/aic_scene/res_pretrain/model'
		self.meta_file = os.path.join(self.checkpoint_path, 'model.ckpt-2426155.meta')
		self.summary_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/model/aic_scene/res_pretrain/log'
		self.pretrained_weight_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/pretrained_model/vgg16_weights.npz'

		self.pretrained_resnet_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/pretrained_model/resnet-pretrained'
		self.pretrained_resnet50_meta = os.path.join(self.pretrained_resnet_path, 'ResNet-L50.meta')
		self.pretrained_resnet152_meta = os.path.join(self.pretrained_resnet_path, 'ResNet-L152.meta')
		self.pretrained_resnet50_ckpt = os.path.join(self.pretrained_resnet_path, 'ResNet-L50.ckpt')
		self.pretrained_resnet152_ckpt = os.path.join(self.pretrained_resnet_path, 'ResNet-L152.ckpt')
