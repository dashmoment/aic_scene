import tensorflow as tf
import numpy as np

import sys
sys.path.append('./utility')
import cifar10
import utility as ut
from sklearn.metrics import confusion_matrix

NUM_CLASSES = 10
cifar10.maybe_download_and_extract()


images_test, cls_test, labels_test = cifar10.load_test_data()

images_test = images_test.astype(np.float32)
labels_test = labels_test.astype(np.int64)

flags = tf.app.flags

flags.DEFINE_string('model_path', './model', 'Directory for saving model')
flags.DEFINE_string('meta_file', 'model/test.ckpt-1500.meta', 'Graph metafile')


FLAGS = flags.FLAGS


def main(argv):
    
    tp = 0
    batch_size = 200

    with tf.Session() as sess:
        
    
        saver = tf.train.import_meta_graph(FLAGS.meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))
        
        graph = tf.get_default_graph()
       
    #    for op in  graph.get_operations(): print(op.name)
       
     
        x = graph.get_tensor_by_name("x:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        dropout = graph.get_tensor_by_name("dropout:0")
        logits = graph.get_tensor_by_name("train/resNet_v1/logits/logits:0")
    
        
        for i in range(len(images_test)//batch_size):
            
            inputs = images_test[i*batch_size:i*batch_size+batch_size,:,:,:]
            labels = labels_test[i*batch_size:i*batch_size+batch_size,:]
              
            
            predict_model = tf.argmax(tf.nn.softmax(logits),1)
            true_label = tf.argmax(labels,1)   
            true_positive = sess.run(ut.clac_accuracy(logits,labels), feed_dict={x:inputs,dropout:1,is_training:False})
            
            tp = tp + true_positive
            
            print("Iteration:{}, Accuracy:{} ".format(i,tp/(i*batch_size+1)))
        


if __name__== '__main__':
    tf.app.run()
        

