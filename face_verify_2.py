from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import detect_face 
from tensorflow.python.platform import gfile

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

predictor_model = "shape_predictor_68_face_landmarks.dat"
detect_face = detect_face.Detect_Face(predictor_model,160)
model = "./Facenet_ckpt/20180402-114759.pb"
image_size = (160, 160)

def predict():
    #Refer predict
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    sess = tf.Session()
    graph = tf.get_default_graph()
    with graph.as_default():
        with sess.as_default(): 
            my_image_paths = ["lfw/Abel_Pacheco/Abel_Pacheco_0001.jpg","lfw/Abel_Pacheco/Abel_Pacheco_0002.jpg","lfw/Alec_Baldwin/Alec_Baldwin_0001.jpg","lfw/Brett_Hull/Brett_Hull_0001.jpg", "lfw/Anthony_Fauci/Anthony_Fauci_0001.jpg","lfw/Anthony_Fauci/Anthony_Fauci_0002.jpg"]       
            load_trained_model(model)
            images = []
            for filename in tf.unstack(my_image_paths):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, 3)
                image.set_shape(image_size + (3,))
                images.append(image)

			
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #my_faces = detect_face.multi_image_detect_face(my_images)
            embeddings = sess.run(embeddings,feed_dict = {images_placeholder:images, phase_train_placeholder: False})
            embeddings1 = embeddings[0::2]
            embeddings2 = embeddings[1::2]
            
            assert(embeddings1.shape[0] == embeddings2.shape[0])
            assert(embeddings1.shape[1] == embeddings2.shape[1])
            mean = 0.00

            dist = distance(embeddings1-mean, embeddings2-mean, distance_metric = 0)
            threshold = 0.7
            predict_issame = np.less(dist, threshold)
            print (predict_issame)

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist


def load_trained_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=None, name='')
    else:
    	saver = tf.train.import_meta_graph("Facenet_ckpt/model.meta")
    	#sess.run(tf.global_variables_initializer())
    	saver.restore(tf.get_default_session(), tf.train.latest_checkpoint("Facenet_ckpt"))

if __name__ == '__main__':
    predict()
#net = cv2.dnn.readNetFromTensorflow('faceNet_graph.pb')
