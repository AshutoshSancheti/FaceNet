from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from Utils import utils
import argparse
import os
import MNet_v2
import detect_face 

tfe = tf.contrib.eager
slim = tf.contrib.slim

saved_model_dir = "saved_model/my_model"  #The protobuf directory
my_ckpt_dir = "mobilenet_v2_checkpoints"
my_checkpoint_file = "mobilenet_v2_checkpoints/my_model"
pretrained_ckpt_dir = "mobilenet_v2/mobilenet_v2"
predictor_model = "shape_predictor_68_face_landmarks.dat"
my_alpha = 0.20
image_size = 224
BATCH_SIZE = 7
is_training = True
LEARNING_RATE = 0.0000005 #0.00000005
learning_rate_decay_epochs = 1
epoch_size = 200
learning_rate_decay_factor = 1.0

#my_input = utils.doub_inputs
my_input = utils.inputs
detect_face = detect_face.Detect_Face(predictor_model,224)


if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','first')):
    os.mkdir(os.path.join('summaries','first'))


def main():
    
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:
            global_step = tf.Variable(0, trainable=False,name='global_step')
            #with tf.name_scope('my_network'):
            x = tf.placeholder(tf.float32, [None, 224, 224, 3],name = 'x_input')
            learning_rate_placeholder = tf.placeholder(dtype = tf.float32,name = 'learning_rate')

            with tf.contrib.slim.arg_scope(MNet_v2.training_scope()):
                logits, endpoints = MNet_v2.mobilenet(x)
                ##### Use this when training for first time ######
                #saver = tf.train.Saver(tf.trainable_variables()) 
                #saver.restore(sess, pretrained_ckpt_dir)

            output_layer = slim.fully_connected(logits, 128, activation_fn = None, trainable = True, scope = 'Encoding_layer',weights_regularizer = slim.l2_regularizer(0.00005))
            my_encodings = tf.nn.l2_normalize(output_layer,axis=0,epsilon=1e-12,name ='my_vector_embeddings')

            #saver = tf.train.Saver() #tf.trainable_variables(), L2_Norm_
            #saver.restore(sess, tf.train.latest_checkpoint(my_ckpt_dir))
            #print ("RESTORING VARIABLES")
        
            learning_rate = tf.train.exponential_decay(0.0000002, global_step,100000, 0.96, staircase=True)
            with tf.name_scope('loss'):
                with tf.name_scope('triplet_loss'):
                    loss = utils.total_triplet_loss_v1(my_encodings)
                    #loss = utils.total_doublet_loss(my_encodings)
                with tf.name_scope('Reg_loss'):
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                #with tf.name_scope('total_loss'): Not required as name='total_loss' already covers it
                total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
                sess.run(tf.local_variables_initializer())
            
            with tf.name_scope('train_step'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate_placeholder) 
                train_op = optimizer.minimize(total_loss, global_step=global_step,name = 'minimize_step')
                sess.run(tf.local_variables_initializer())

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)

            tf_loss_summary = tf.summary.scalar('total_loss', total_loss)
            merged = tf.summary.merge_all()
            summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)          

            # Put this after the init_op not before otherwise it will reinitialize all variables randomly 
            saver = tf.train.Saver(max_to_keep = 2)
            saver.restore(sess, tf.train.latest_checkpoint(my_ckpt_dir))
            print ("RESTORING VARIABLES")

            #Creates the serialized protocol buffer '.pb' file
            #saved_model = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
            #saved_model.add_meta_graph_and_variables(sess,tags = ['Mnet_face'])

            LEARNING_RATE = 0.00000002
            for batch1 in my_input(BATCH_SIZE):
            #for i in range(epoch_size):
                #batch1 = my_input.__next__()
                if(len(batch1)!= (3*BATCH_SIZE)):
                    continue
                batch = detect_face.multi_image_detect_face(batch1)
                j = 0
                while(j<20):
                    _, tota_loss, summary,my_global_step, lr = sess.run([train_op, loss,merged,global_step,learning_rate], feed_dict = {x: batch, learning_rate_placeholder:LEARNING_RATE})
                    print ("EPOCH NO. ",my_global_step)
                    print ("TOTAL LOSS:", tota_loss)
                    print ("LEARNING RATE:", lr)
                    summ_writer.add_summary(summary, global_step.eval())
                    j = j + 1
                    LEARNING_RATE = (LEARNING_RATE * 0.99)
                saver.save(sess, my_checkpoint_file, global_step = global_step)
                #saved_model.save()




if __name__ == '__main__':
    main()











#with tf.variable_scope('MobilenetV2','MobilenetV2',[x]) as scope:  
            #This line is resulting in scope becoming /MobilenetV2/MobilenetV2/... while in the ckpt file the variable scope are /MobilenetV2/... i.e. this tf.variable_scope is adding another /MobilenetV2/  along with the tf.contrib.slim.arg_scope() and MNet_v2.mobilenet(x, scope = 'MobilenetV2')
