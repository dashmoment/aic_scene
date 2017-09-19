import tensorflow as tf

import sys
sys.path.append('./utility')
import config
import data_utility as du
import netfactory as nf

conf = config.config('home')
data_u = du.data_utility("home")

initializer=tf.contrib.layers.xavier_initializer()
tp = 0




with tf.Session() as sess:
        
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(conf.pretrained_resnet50_meta)
    saver.restore(sess, conf.pretrained_resnet50_ckpt)
        
    graph = tf.get_default_graph()      
#    for op in  graph.get_operations(): print(op.name)
	
    try:
        weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
        bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)
    except:
        scope.reuse_variables()
        weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
        bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)

    init_new_vars_op = tf.initialize_variables([weight, bias])
    sess.run(init_new_vars_op)
    test_batch = data_u.get_batch('validation', conf.validation_idx_list, 0)
    is_training = graph.get_tensor_by_name("plcaholders/is_training:0")
    images = graph.get_tensor_by_name("images:0")
    
    net = graph.get_tensor_by_name("avg_pool:0")
    logits = tf.nn.xw_plus_b(net, weight, bias, name="fc3")
#	x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    netout = sess.run(logits, feed_dict = {images:test_batch[0]})

    print(netout.shape)
#    accurate = tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(predict,1), conf.label), tf.int64))#

#    for i in range(len(conf.validation_idx_list)//batch_size):#

#        print("Iteration:{}/{}".format(i,len(conf.validation_idx_list)))
#        
#        test_batch = data_u.get_batch('validation', conf.validation_idx_list, i)
#        feed_dict = {imgs:test_batch[0],conf.label:test_batch[1], conf.learning_rate:1e-4, conf.is_training:False, conf.dropout:1}
#        acc_num= sess.run(accurate, feed_dict = feed_dict)
#        
#        tp = tp + acc_num#

#    print("Accuracy: ", tp/len(conf.validation_idx_list))



