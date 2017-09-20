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

    
    is_training = graph.get_tensor_by_name("plcaholders/is_training:0")
    images = graph.get_tensor_by_name("images:0")
    net = graph.get_tensor_by_name("avg_pool:0")
	
    try:
        weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
        bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)
    except:
        scope.reuse_variables()
        weight = tf.get_variable("weights",[2048, 80], tf.float32, initializer=initializer)
        bias = tf.get_variable("bias",[80], tf.float32, initializer=initializer)

    with tf.variable_scope('preprocess') as scope:
        mean = tf.constant([114.156, 121.907, 126.488], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        processed_img = conf.imgs-mean

    with tf.variable_scope("network"):
        logits = tf.nn.xw_plus_b(net, weight, bias, name="fc3")
        

    with tf.variable_scope("training"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = conf.label, logits=logits, name="loss"))
        #train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss)


    init_new_vars_op = tf.initialize_variables([weight, bias])
    sess.run(init_new_vars_op)
    
    
    train_batch = data_u.get_batch('validation', conf.validation_idx_list, 0)
    pimg = sess.run(processed_img, feed_dict = {conf.imgs:train_batch[0]})

    feed_dict = {images:pimg, conf.label:train_batch[1], conf.learning_rate:1e-4, conf.is_training:True, conf.dropout:0.5}

    netout, l = sess.run([net,loss], feed_dict = feed_dict)

    print(netout, pimg[0],l, len(train_batch[1]) )

    import cv2
    cv2.imshow("raw", train_batch[0][0])
    cv2.imshow("test", pimg[0])
    cv2.waitKey()


