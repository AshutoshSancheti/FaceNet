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

slim = tf.contrib.slim

my_ckpt_dir = "mobilenet_v2_checkpoints"
my_checkpoint_file = "mobilenet_v2_checkpoints/my_model"

detect_face = detect_face.Detect_Face(predictor_model,224)
#my_batch = ["",""]
#my_batch = np.char.array(my_batch)
#batch = detect_face.multi_image_detect_face(my_batch)

def predict():
    #Refer predict
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_trained_model()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("x_input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("Embeddings:0")


def load_trained_model():
    #Refer facenet for errors
    files = os.listdir(my_ckpt_dir)
    meta = [s for s in files if s.endswith('.meta')] #Extract all .meta files
    meta_file = meta[0]
    saver = tf.train.import_meta_graph(os.path.join(my_ckpt_dir, "my_model-1100.meta"))
    saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(my_ckpt_dir))


predict()
