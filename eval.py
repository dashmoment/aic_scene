import tensorflow as tf

import sys
sys.path.append('./utility')
import config
import data_utility as du

conf = config.config('home')
data_u = du.data_utility("home")

   
tp = 0
batch_size = 200

with tf.Session() as sess:
        
    
    saver = tf.train.import_meta_graph(conf.meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(conf.checkpoint_path))
        
    graph = tf.get_default_graph()
       
    for op in  graph.get_operations(): print(op.name)

    #imgs = graph.get_tensor_by_name("plcaholders/images_1:0")
#    predict = graph.get_tensor_by_name("vgg16/Softmax:0")
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



