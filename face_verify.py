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
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

slim = tf.contrib.slim

my_ckpt_dir = "mobilenet_v2_checkpoints"
predictor_model = "shape_predictor_68_face_landmarks.dat"

detect_face = detect_face.Detect_Face(predictor_model,224)
#my_batch = ["",""]
#my_batch = np.char.array(my_batch)
#batch = detect_face.multi_image_detect_face(my_batch)

def predict():
    #Refer predict
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    sess = tf.Session()
    graph = tf.get_default_graph()
    with graph.as_default():
        with sess.as_default():        
            load_trained_model(sess)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("x_input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("my_vector_embeddings:0")
            my_images = ["lfw/Bill_Gates/Bill_Gates_0001.jpg","lfw/Bill_Gates/Bill_Gates_0002.jpg","lfw/Aaron_Sorkin/Aaron_Sorkin_0001.jpg","lfw/Aaron_Sorkin/Aaron_Sorkin_0002.jpg"]
            my_faces = detect_face.multi_image_detect_face(my_images)
            embeddings = sess.run(embeddings,feed_dict = {images_placeholder:my_faces})
            print (embeddings[0])
            print (embeddings[0].shape)
            print ("Same Person: ",np.sum((np.square(embeddings[0],embeddings[1]))))
            #print (np.sum(np.square(np.subtract(embeddings[0],embeddings[1]))))
            print ("Same Person: ",np.sum(np.square(np.subtract(embeddings[2],embeddings[3]))))
            print ("Differnet: ",np.sum(np.square(np.subtract(embeddings[0],embeddings[2]))))
            print ("Different: ",np.sum(np.square(np.subtract(embeddings[2],embeddings[1]))))


def load_trained_model(sess):
    #Refer facenet for errors
    files = os.listdir(my_ckpt_dir)
    meta = [s for s in files if s.endswith('.meta')] #Extract all .meta files
    meta_file = meta[0]
    saver = tf.train.import_meta_graph(os.path.join(my_ckpt_dir, "my_model-1100.meta"))
    sess.run(tf.global_variables_initializer())
    saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(my_ckpt_dir))

#if __name__ == '__main__':
#    predict()
net = cv2.dnn.readNetFromTensorflow('faceNet_graph.pb')
