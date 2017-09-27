import tensorflow as tf

import sys
sys.path.append('./utility')
import config
import data_utility as du
import netfactory as nf
import log_utility as log_u
import resnet as rnet

conf = config.config('office')
data_u = du.data_utility("office")

#initializer=tf.contrib.layers.xavier_initializer()
initializer = tf.truncated_normal_initializer(stddev = 0.001)
tp = 0

saver = tf.train.import_meta_graph(conf.pretrained_resnet50_meta)
graph = tf.get_default_graph()   
is_training = graph.get_tensor_by_name("plcaholders/is_training:0")
images = graph.get_tensor_by_name("images:0")
net = graph.get_tensor_by_name("avg_pool:0")
#net2 =  graph.get_tensor_by_name('scale1/weights:0')

with tf.name_scope('preprocess') as scope:
    mean = tf.constant([114.156, 121.907, 126.488], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    processed_img = conf.imgs-mean

#_, net  = rnet.inference(images, True) 

net = tf.stop_gradient(net)



var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for op in var_list: print(op.name)


with tf.Session() as sess:
    
    saver = tf.train.Saver(var_list)
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess, conf.pretrained_resnet50_ckpt)
        
      
#    for op in  graph.get_operations(): print(op.name)
    

#    tf.control_dependencies()
    try:
        with tf.variable_scope("aic_fc"):
            weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)
    except:
            with tf.variable_scope("aic_fc") as scope:
                scope.reuse_variables()
                weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
                bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)
    
    with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([114.156, 121.907, 126.488], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            processed_img = conf.imgs-mean
    
    with tf.variable_scope("network"):
            logits = tf.nn.xw_plus_b(net, weight, bias, name="fc3")
    
    
#var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#removed = []
#for idx in  range(len(var_list)): 
#        if var_list[idx].name ==  'fc/weights:0' or   var_list[idx].name == "fc/biases:0":
#            removed.append(var_list[idx])
#            
#    var_list.remove(removed[0])
#    var_list.remove(removed[1])
    
    with tf.variable_scope("training"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = conf.label, logits=logits, name="cross_entropy"), name="loss")
            opt = tf.train.AdamOptimizer(conf.learning_rate)
            train_op = opt.compute_gradients(loss)
            #train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)
	
    
            
    init_new_vars_op = tf.variables_initializer([weight, bias])
    sess.run(init_new_vars_op,)
    
      
    
    log = log_u.logger(sess, checkpoint_path=conf.checkpoint_path, summary_path=conf.summary_path)
    
    train_batch = data_u.get_batch('validation', conf.validation_idx_list, 0)
    
    pimg = sess.run(processed_img, feed_dict = {conf.imgs:train_batch[0]})

    print(pimg.shape)

    feed_dict = {images:pimg, conf.label:train_batch[1], conf.learning_rate:1e-4, conf.is_training:True, conf.dropout:0.5}
    
    netout, _ = sess.run([loss  ,train_op], feed_dict = feed_dict)
    print(netout)
   

   


