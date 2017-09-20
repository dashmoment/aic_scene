import tensorflow as tf
import cv2
import sys
sys.path.append('./utility')
import data_utility
import config
import data_utility as du

conf = config.config('office')
data_u = du.data_utility("office")

   
tp = 0
ensemble = False

with tf.Session() as sess:
        
    
    saver = tf.train.import_meta_graph(conf.meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(conf.checkpoint_path))
        
    graph = tf.get_default_graph()
       
    #for op in  graph.get_operations(): print(op.name)
    imgs = graph.get_tensor_by_name("plcaholders/images_1:0")
    

    if ensemble:
        logits = graph.get_tensor_by_name("vgg16/vgg16_aic_v1/fc3_aic/fc3_aic:0")
        esemble_logits = tf.reduce_mean(logits,0)
        predict = tf.nn.softmax(esemble_logits)
        esemble_predict = tf.reduce_mean(predict,0)
        accurate = tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(predict,0), conf.label), tf.int64))

    
    else:
        predict = graph.get_tensor_by_name("vgg16/Softmax:0")
        accurate = tf.reduce_sum(tf.cast(tf.equal(tf.arg_max(predict,1), conf.label), tf.int64))

        
   
    for i in range(len(conf.validation_idx_list)//conf.test_batch_size):

        print("Iteration:{}/{}".format(i,len(conf.validation_idx_list)))
        
        if ensemble:
            test_batch = data_u.get_ensemble_batch(conf.validation_idx_list, i)
        else:
            test_batch = data_u.get_batch('validation', conf.validation_idx_list, i)

        
        feed_dict = {imgs:test_batch[0],conf.label:test_batch[1], conf.learning_rate:1e-4, conf.is_training:False, conf.dropout:1}
        acc_num= sess.run(accurate, feed_dict = feed_dict)
            
        tp = tp + acc_num

    print("Accuracy: ", tp/len(conf.validation_idx_list))



