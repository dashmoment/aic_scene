import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


import os 
import sys
sys.path.append('./utility')
import cifar10
import utility as ut

import help_function as h
import model_zoo

flags = tf.app.flags

flags.DEFINE_string('save_root', './', 'Directory for saving model & log')
flags.DEFINE_string('checkpoint_name', './test.ckpt', 'Name for checkpoint file')
flags.DEFINE_integer('batch_Size', 128, 'Batch_Size')
flags.DEFINE_integer('iteration', 282030, 'start step')
flags.DEFINE_boolean('restore_training', True, 'If restore traing')

FLAGS = flags.FLAGS


checkpoint_dir = os.path.join(FLAGS.save_root, 'model') 
log_dir = os.path.join(FLAGS.save_root, 'log') 

if not os.path.isdir(FLAGS.save_root): os.mkdir(FLAGS.save_root)
if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
if not os.path.isdir(log_dir): os.mkdir(log_dir)


cifar10.maybe_download_and_extract()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
images_train.astype(np.float32)
labels_train.astype(np.int64)
images_test.astype(np.float32)
labels_test.astype(np.int64)

from cifar10 import img_size, num_channels, num_classes
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
is_training = tf.placeholder(tf.bool, name='is_training')
dropout = tf.placeholder(tf.float32, name='dropout')


mz = model_zoo.model_zoo(x, dropout,is_training, "resNet_v1")

with tf.name_scope("train"):
    
    logits = mz.build_model()
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)  
    loss = tf.reduce_mean(losses, name='loss')        
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)    
    accuracy = ut.clac_accuracy(logits, y_true)


with tf.name_scope('train_summary'):
    tf.summary.scalar("Cross_Entropy", loss, collections=['train'])
    tf.summary.scalar("True_Positive", accuracy, collections=['train'])
    merged_summary_train = tf.summary.merge_all('train') 
    
with tf.name_scope('test_summary'):
    tf.summary.scalar("Cross_Entropy", loss, collections=['test'])
    tf.summary.scalar("True_Positive", accuracy, collections=['test'])
    merged_summary_test = tf.summary.merge_all('test') 
    

def main(argv):

 
    with tf.Session() as sess:
        
        
        saver = tf.train.Saver() 
        summary_writer = tf.summary.FileWriter('log', sess.graph)    
        sess.run(tf.global_variables_initializer())
        
        if FLAGS.restore_training == True:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
           
        
    #    a = sess.run(logits, feed_dict = {x:images_train[0:5,:,:,:], dropout:0.5,is_training:True})
        
        
        
        while True:
            
            datapool = ut.ramdom_batch(len(images_train), FLAGS.batch_Size)
            
            for i in range(len(datapool)):
            
            
                FLAGS.iteration = FLAGS.iteration + 1         
                x_batch = images_train[datapool[i]]
                y_true_batch = labels_train[datapool[i]]  
                x_testbatch, y_test_batch = h.random_batch(images_test, labels_test, FLAGS.batch_Size)
        
           
        
                train_loss, acc,  train_sum,_ = sess.run([loss, accuracy,merged_summary_train,optimizer], feed_dict = {x:x_batch,y_true:y_true_batch,dropout:0.5,is_training:True})
                test_loss, tacc, test_sum = sess.run([loss, accuracy,merged_summary_test], feed_dict = {x:x_testbatch,y_true:y_test_batch,dropout:1,is_training:False})
                
                
                if FLAGS.iteration%500 == 0:
                    summary_writer.add_summary(train_sum, FLAGS.iteration)
                    summary_writer.add_summary(test_sum, FLAGS.iteration)
                    saver.save(sess, os.path.join(checkpoint_dir, FLAGS.checkpoint_name), global_step=FLAGS.iteration)
               
                
                print("Step:{}, Train loss:{}, accuracy:{}".format(FLAGS.iteration, train_loss, acc))
                print("Test loss:{}, accuracy:{}".format(test_loss, tacc))

      
if __name__=="__main__":
    
    tf.app.run()
    

    
    
    
    
    
    
    
    
    
    
    