import tensorflow as tf
import netfactory as nf
import numpy as np

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training
        
    
        
    def googleLeNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "conv2": [3,3,128],
            "inception_1":{                 
                    "1x1":64,
                    "3x3":{ "1x1":96,
                            "3x3":128
                            },
                    "5x5":{ "1x1":16,
                            "5x5":32
                            },
                    "s1x1":32
                    },
            "inception_2":{                 
                    "1x1":128,
                    "3x3":{ "1x1":128,
                            "3x3":192
                            },
                    "5x5":{ "1x1":32,
                            "5x5":96
                            },
                    "s1x1":64
                    },
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("googleLeNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1],name="conv2", flatten=False)
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.inception_v1(net, model_params, name= "inception_1", flatten=False)
            net = nf.inception_v1(net, model_params, name= "inception_2", flatten=False)
            net = tf.nn.avg_pool (net, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='VALID')
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits
    
    
    def resNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "rb1_1": [3,3,64],
            "rb1_2": [3,3,64],
            "rb2_1": [3,3,128],
            "rb2_2": [3,3,128],
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("resNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            id_rb1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            
            net = nf.convolution_layer(id_rb1, model_params["rb1_1"], [1,1,1,1],name="rb1_1")
            id_rb2 = nf.convolution_layer(net, model_params["rb1_2"], [1,1,1,1],name="rb1_2")
            
            id_rb2 = nf.shortcut(id_rb2,id_rb1, name="rb1")
            
            net = nf.convolution_layer(id_rb2, model_params["rb2_1"], [1,2,2,1],padding="SAME",name="rb2_1")
            id_rb3 = nf.convolution_layer(net, model_params["rb2_2"], [1,1,1,1],name="rb2_2")
            
            id_rb3 = nf.shortcut(id_rb3,id_rb2, name="rb2")
            
            net  = nf.global_avg_pooling(id_rb3, flatten=True)
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits
    
    
    def vgg16_aic_v1(self):
        
        model_params = {
        
            "conv1_1": [3,3, 64],
            "conv1_2": [3,3,64],
            "conv2_1": [3,3,128],
            "conv2_2": [3,3,128],
            "conv3_1": [3,3,256],
            "conv3_2": [3,3,256],
            "conv3_3": [3,3,256],
            "conv4_1": [3,3,512],
            "conv4_2": [3,3,512],
            "conv4_3": [3,3,512],
            "conv5_1": [3,3,512],
            "conv5_2": [3,3,512],
            "conv5_3": [3,3,512],
            "fc1": 4096,
            "fc2": 4096,
            "fc3": 1000,
                     
        }
        
        with tf.name_scope("vgg16_aic_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1_1"], [1,1,1,1],name="conv1_1")
            net = nf.convolution_layer(net, model_params["conv1_2"], [1,1,1,1],name="conv1_2")
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            net = nf.convolution_layer(net, model_params["conv2_1"], [1,1,1,1],name="conv2_1")
            net = nf.convolution_layer(net, model_params["conv2_2"], [1,1,1,1],name="conv2_2")
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            net = nf.convolution_layer(net, model_params["conv3_1"], [1,1,1,1],name="conv3_1")
            net = nf.convolution_layer(net, model_params["conv3_2"], [1,1,1,1],name="conv3_2")
            net = nf.convolution_layer(net, model_params["conv3_3"], [1,1,1,1],name="conv3_3")
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            net = nf.convolution_layer(net, model_params["conv4_1"], [1,1,1,1],name="conv4_1")
            net = nf.convolution_layer(net, model_params["conv4_2"], [1,1,1,1],name="conv4_2")
            net = nf.convolution_layer(net, model_params["conv4_3"], [1,1,1,1],name="conv4_3")
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool4')
            net = nf.convolution_layer(net, model_params["conv5_1"], [1,1,1,1],name="conv5_1")
            net = nf.convolution_layer(net, model_params["conv5_2"], [1,1,1,1],name="conv5_2")
            net = nf.convolution_layer(net, model_params["conv5_3"], [1,1,1,1],name="conv5_3")
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool5')
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
            net = nf.fc_layer(net, model_params["fc1"], name="fc1")
            net = nf.fc_layer(net, model_params["fc2"], name="fc2")
            logits = nf.fc_layer(net, model_params["fc3"], name="fc3", activat_fn=None)

        return logits
    
    def build_model(self):
        model_list = ["googleLeNet_v1", "resNet_v1", "vgg16_aic_v1"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, self.model_ticket)
            netowrk = fn()
            return netowrk
        
        
def unit_test(input_shapes = [None, 32, 32, 3]):

    x = tf.placeholder(tf.float32, shape=input_shapes, name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"vgg16_aic_v1")
    return mz.build_model()
    

#m = unit_test([None, 244, 244, 3])