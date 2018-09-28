from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Doing this as ros installation is causing an error, there are permanent solutions but may interfer with ros
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove()
import cv2

import tensorflow as tf
import numpy as np
import pandas as pd
import inception_blocks_v4
import inception_utils
import os
import random  #import VGG16
import detect_face

slim = tf.contrib.slim

train_dir = "lfw"
pretrained_ckpt_dir = "checkpoint_file/inception_v4.ckpt"
my_ckpt_dir = "my_checkpoints"
my_alpha = 0.2
image_size = 299
batch_size = 8

predictor_model = "shape_predictor_68_face_landmarks.dat"
test = detect_face.Detect_Face(predictor_model)

is_training = True

def triplet_loss(anchor, positive, negative, alpha = my_alpha):
    """
    anchor, postive, negative are n-dimensional vectors (tensors)
    alpha - the difference b/w anchor positive and anchor negative pairs; A tensor;
    """
    delta_1 = tf.reduce_sum(tf.square(anchor - positive))  #square of euclidean distance or L2 norm
    delta_2 = tf.reduce_sum(tf.square(anchor - negative))
    loss = tf.maximum(delta_1 - delta_2 + alpha, tf.constant(0, dtype = tf.float32))
    return loss

def total_triplet_loss(my_list):
    #My list is of the form [3*batch_size, 128] - 1st represent anchor, 2nd-positive, 3rd-    		negative, 4th-anchor, 5th-positive
    total_loss = 0
    for i in range(batch_size):
        total_loss += triplet_loss(my_list[i+0],my_list[i+1],my_list[i+2]) #i = i + 2
    return total_loss

### THIS MODEL IS JUST FOR TRAINING THE SIAMESE NETWORK ### CHECK WHETHER THE RETURN STATEMENT IS CORRECT
def model_image_encoding(X_input_string, sess, first_training_step = not (bool(tf.train.latest_checkpoint(my_ckpt_dir)))):
    with tf.device('/device:CPU:0'):
        with slim.arg_scope(inception_blocks_v4.inception_v4_arg_scope()):
            ############ 3 placeholders for each image ################
            #X_input_anchor_string = tf.placeholder(tf.string, [None,1])
            #X_input_positive_string = tf.placeholder(tf.string, [None,1])
            #X_input_negative_string = tf.placeholder(tf.string, [None,1])
            print ("*************************************************************************************************************************")
            X_input = read_image_decode(X_input_string,sess=sess)
            
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)
            print ("START OF ARCHITECTURE")
            logits, end_points = inception_blocks_v4.inception_v4(inputs = X_input, num_classes = 1001, is_training=True,dropout_keep_prob = 0.8, reuse=None, scope = 'InceptionV4' , create_aux_logits= False)
            
            ################ USE THIS WHEN TRAINING FOR FIRST TIME #############################
            #pretrained_weights = slim.assign_from_checkpoint_fn(pretrained_ckpt_dir, slim.get_model_variables('InceptionV4'))
            #pretrained_weights(sess)
            
            #ADD AN ENCODING LAYER
            my_encodings = slim.fully_connected(logits, 128, activation_fn = tf.nn.relu, trainable = is_training, scope = 'Encoding_layer',weights_initializer = tf.truncated_normal_initializer(stddev = 0.001), weights_regularizer = slim.l2_regularizer(0.00005),      biases_initializer = tf.zeros_initializer(), biases_regularizer = None)
        
            ################## USE THIS WHEN YOU HAVE YOUR OWN CHECKPOINT #######################
            saver = tf.train.Saver() #saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            saver.restore(sess, tf.train.latest_checkpoint(my_ckpt_dir))
            print ("RESTORING VARIABLES")
            print ("END OF ARCHITECTURE")
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)
            return my_encodings


def train_op(sess):
    """
    image_triplets: A list of tuples(a batch) with each tuple containing an anchor, positive, negative image
    """
    #my_embed_list = []
    #image_triplets_string = inputs()
    X_input_string = tf.placeholder(tf.string, [None,1])
    my_embed_list = model_image_encoding(X_input_string,sess = sess)
    loss = total_triplet_loss(my_embed_list)
    print ("CALCULATED LOSS")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) 
    train = optimizer.minimize(loss)
    print ("MY OPTIMIZE STEP IS DONE")
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    return train


def get_dataset(input_directory):
    dataset = []
    classes = os.listdir(input_directory)  #This gives a list of all the folder names in the input_directory
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(input_directory, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir) #list of image names in each "classes" directory. len(images) can give me whether there are more than 1 image for each class.
            image_paths = [os.path.join(facedir, img) for img in images]
            dataset.append((class_name, image_paths))
            #This will give me tuple where 1 element is the class_name and second element is the list of all image paths of that class os person
    return dataset

def read_image_decode(input_strings,sess, random_flip = True, random_brightness = True, random_contrast = True):
    images = []
    print ("*************************************************************************************************************************")
    for i in range((3*batch_size)):
        file_contents = tf.read_file(input_strings[i][0])
        print ("*************************************************************************************************************************")
        image = tf.image.decode_jpeg(file_contents, channels=3)
        image = tf.image.convert_image_dtype(image, dtype = tf.float32)
        print ("**********************************************$$$$$$$$$$*****************************************************************")
        #image.eval(session = tf.Session()) this makes tensor->numpy array
        #image = cv2.imread(input_strings[i][0])
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        #with tf.Session() as default:
         #   image = test.detect_face(image.eval()) # thsi is causing error
        print ("************************************************###############************************************************************")
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        test_dim = tf.constant(299, dtype = tf.int32)
        height_less = tf.less(height, 299)
        width_less = tf.less(width, 299)
        resize_dimensions = tf.constant([image_size, image_size], dtype = tf.int32)
        image = tf.cond(height_less, lambda: tf.image.resize_images(image,resize_dimensions), lambda: image)
        image = tf.cond(width_less, lambda: tf.image.resize_images(image,resize_dimensions), lambda: image)
        
        image = tf.random_crop(image, size=[image_size, image_size, 3])
       
        image = tf.image.per_image_standardization(image)
        if random_flip:
            image = tf.image.random_flip_left_right(image)

        if random_brightness:
            image = tf.image.random_brightness(image, max_delta=0.3)

        if random_contrast:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        #image = tf.expand_dims(image, 0)    #Convert image from [299,299,3] to [1,299,299,3]
        images.append(image)
    stack_images = tf.stack(images, axis = 0) #check tf.concat
    #tf.stack is tf.expand_dims and tf.concat in one
    print ("GENERATED THE IMAGES FROM STRINGS")
    return stack_images

"""
def read_image_decode(input_strings, random_flip = True, random_brightness = True, random_contrast = True):

    images = []
    for i in range((3*batch_size)):
        #file_contents = tf.read_file(input_strings[i])
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        #image.eval(session = tf.Session()) this makes tensor->numpy array
        image = cv2.imread(input_strings[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = test.detect_face(image) 
        image = tf.image.convert_image_dtype(image, dtype = tf.float32)
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        test_dim = tf.constant(299, dtype = tf.int32)
        height_less = tf.less(height, 299)
        width_less = tf.less(width, 299)
        resize_dimensions = tf.constant([image_size, image_size], dtype = tf.int32)
        image = tf.cond(height_less, lambda: tf.image.resize_images(image,resize_dimensions), lambda: image)
        image = tf.cond(width_less, lambda: tf.image.resize_images(image,resize_dimensions), lambda: image)
        
        image = tf.random_crop(image, size=[image_size, image_size, 3])
       
        image = tf.image.per_image_standardization(image)
        if random_flip:
            image = tf.image.random_flip_left_right(image)

        if random_brightness:
            image = tf.image.random_brightness(image, max_delta=0.3)

        if random_contrast:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        #image = tf.expand_dims(image, 0)    #Convert image from [299,299,3] to [1,299,299,3]
        images.append(image)
    stack_images = tf.stack(images, axis = 0) #check tf.concat
    #tf.stack is tf.expand_dims and tf.concat in one
    print ("GENERATED THE IMAGES FROM STRINGS")
    return stack_images 
"""

def inputs():
    """
    generate a batch or list with each element a tuple of 3 images: anchor, postive and negative
    use yield statement or the tf.train.batch_join api
    use get_dataset function.
    sort them into lists having 1 or more than 1 images etc.
    Then use these lists as inputs into the batch api
    """
    #sorting part
    one_image = []
    more_than_one_image = []
    dataset = get_dataset(train_dir)
    for image_path_set in dataset:
        if (len(image_path_set[1]) == 1):
            one_image.append(image_path_set[1])  #appending only the image paths
        else:
            more_than_one_image.append(image_path_set[1])
    
    while True:
        my_batch = []
        for i in range(batch_size):
            negative_string = (random.sample(one_image, 1))[0][0]  #[0][0] used because random.sample give a list with of 1 list as it's element. Just wanted to get the string not as list of string
            
            anchor_positive_string_set = random.sample(more_than_one_image, 1)
            #getting one class image paths in the form a lsit of strings which inside another list - [['string1', 'string2', ..]]
            anchor_positive_string = random.sample(anchor_positive_string_set[0],2)
            #taking any two strings from the 1st element(list) of the outer list. output = ['anchor_string', 'postive_string']
            anchor_string = anchor_positive_string[0]
            positive_string = anchor_positive_string[1]
            #print (anchor_string, positive_string, negative_string) #testing purpose
         
            my_batch.append(anchor_string)
            my_batch.append(positive_string)
            my_batch.append(negative_string)
        print ("OUTPUTING A BATCH. THIS OUTPUTS A 2-D  NUMPY CHAR ARRAY OF SIZE (BATCH_SIZE*3)")
        my_batch = np.char.array(my_batch)
        yield my_batch
        
