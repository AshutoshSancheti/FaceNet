from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Doing this as ros installation is causing an error, there are permanent solutions but may interfer with ros
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random  

slim = tf.contrib.slim

train_dir = "lfw"
my_alpha = 0.2
image_size = 224
BATCH_SIZE = 7


def inputs(batch_size):
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


def doub_inputs(batch_size):
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
            pass
        else:
            more_than_one_image.append(image_path_set[1])

    while True:
        my_batch = []
        for i in range(batch_size):
            anchor_positive_string_set = random.sample(more_than_one_image, 1)
            anchor_positive_string = random.sample(anchor_positive_string_set[0],2)
            anchor_string = anchor_positive_string[0]
            positive_string = anchor_positive_string[1]
            my_batch.append(anchor_string)
            my_batch.append(positive_string)
        
        print ("OUTPUTING A BATCH. THIS OUTPUTS A 2-D  NUMPY CHAR ARRAY OF SIZE (BATCH_SIZE*2)")
        my_batch = np.char.array(my_batch)
        yield my_batch
         

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


def triplet_loss(anchor, positive, negative, alpha = my_alpha):
    """
    anchor, postive, negative are n-dimensional vectors (tensors)
    alpha - the difference b/w anchor positive and anchor negative pairs; A tensor;
    """
    with tf.name_scope('single_trip_loss'):
        delta_1 = tf.square(tf.subtract(anchor, positive))  #square of euclidean distance or L2 norm
        delta_2 = tf.square(tf.subtract(anchor, negative))
        loss = tf.maximum(tf.reduce_sum(tf.square(tf.subtract(delta_1, delta_2))), tf.constant(0, dtype = tf.float32))
    return loss


def total_triplet_loss_v1(my_encodings):
    my_encodings = tf.reshape(my_encodings, [3,BATCH_SIZE, 128])
    my_encodings = tf.unstack(my_encodings)
    delta_1 = tf.reduce_sum(tf.square(tf.subtract(my_encodings[0], my_encodings[1])), 1)
    delta_2 = tf.reduce_sum(tf.square(tf.subtract(my_encodings[0], my_encodings[2])), 1)
        
    init_loss = tf.add(tf.subtract(delta_1,delta_2), my_alpha)
    total_loss = tf.reduce_mean(tf.maximum(init_loss, 0), 0)
    return total_loss


def total_triplet_loss_v0(my_encodings,sess = None):
    #my_triplets = tf.unstack(my_encodings)   #This converts [None,128] tensor into list of [128] shaped tensors and we can iterate over this list now
    total_loss = tf.constant(0,dtype = tf.float32, name = 'initial_trip_loss')
    for i in range(BATCH_SIZE):
        total_loss += tf.add(triplet_loss(my_encodings[(3*i)+0],my_encodings[(3*i)+1],my_encodings[(3*i)+2]),total_loss)
    return total_loss


############################ ERROR IN GENERATING THE TRIPLETS ################################3
def total_triplet_loss_v2(my_encodings,sess):
    #my_triplets = tf.unstack(my_list)   #This converts [None,128] tensor into list of [128] shaped tensors and we can iterate over this list now
    shape = [(2*BATCH_SIZE),128]
    assert_tensor_shape(my_encodings, shape, "my_encodings")
    total_loss = tf.constant(0,dtype = tf.float32, name = 'initial_trip_loss') 
    my_triplets = generate_triplets(my_encodings)
    my_triplet_list = tf.unstack(my_triplets)
    #t = my_triplets.get_shape().as_list()
    for my_triplet in my_triplet_list:
         total_loss += tf.add(triplet_loss(my_triplet[0],my_triplet[1],my_triplet[2]),total_loss)       
    return total_loss


def generate_triplets(my_encodings, sess=None):
    my_triplets = tf.constant(0,tf.float32, shape = [1,3,128]) #INITIAL TENSOR FOR APPENDING REMOVED LATER

    for index in range(0,(2*BATCH_SIZE),2):
        for t_index in range(2*BATCH_SIZE):
            if((t_index == index) or (t_index == (index+1))):
                continue
            
            else:
                check = tf.less_equal(tf.reduce_sum(tf.square(tf.subtract(my_encodings[index], my_encodings[t_index]))),24000)
                my_triplet = tf.stack([my_encodings[index], my_encodings[index+1], my_encodings[t_index]],axis = 0)
                # my_triplet = tf.concat([tf.expand_dims(my_encodings[index], axis = 0),tf.expand_dims(my_encodings[index+1], axis = 0), tf.expand_dims(my_encodings[t_index], axis = 0)
                #tf.stack expands the dimensions from [128]->[1,128] and then concats thems to [3,128]
                append_triplet = tf.concat([my_triplets, tf.expand_dims(my_triplet, axis = 0)], axis = 0)
                my_triplets = tf.cond(check, lambda: append_triplet , lambda: my_triplets, name = 'trip_assign_one')
    #my_triplets = tf.stack(my_triplets)  #convert the list into a single tensor
    if sess is not None:
        print('genfunc', sess.run(my_triplets))
    return my_triplets


def doublet_loss(anchor, positive):
    """
    anchor, postive, negative are n-dimensional vectors (tensors)
    alpha - the difference b/w anchor positive and anchor negative pairs; A tensor;
    """
    loss = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), name = 'single_trip_loss')  #square of euclidean distance or L2 norm
    #delta_2 = tf.reduce_sum(tf.square(anchor - negative))
    #loss = tf.maximum(delta_1, tf.constant(0, dtype = tf.float32))
    return loss


def total_doublet_loss(my_encodings):
    #My list is of the form [3*batch_size, 128] - 1st represent anchor, 2nd-positive, 3rd-    		negative, 4th-anchor, 5th-positive
    #my_encodings = tf.unstack(my_encodings)
    my_encodings = tf.reshape(my_encodings, [2,BATCH_SIZE, 128])
    #total_loss = tf.constant(0,dtype = tf.float32, name = 'initial_trip_loss')
    total_loss = tf.reduce_sum(tf.square(tf.subtract(my_encodings[0],my_encodings[1])), axis = None)
    #for i in range(BATCH_SIZE):
        #total_loss += doublet_loss(my_encodings[(2*i)+0],my_encodings[(2*i)+1]) 
    return total_loss


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


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
        image = tf.image.convert_image_dtype(input_image, dtype = tf.float32)
       
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
    


def assert_tensor_shape(tensor, shape, display_name):
    assert tf.assert_rank(tensor, len(shape), message='{} has wrong rank'.format(display_name))

    tensor_shape = tensor.get_shape().as_list() if len(shape) else []

    wrong_dimension = [ten_dim for ten_dim, cor_dim in zip(tensor_shape, shape)
                       if cor_dim is not None and ten_dim != cor_dim]
    assert not wrong_dimension, \
        '{} has wrong shape.  Found {}'.format(display_name, tensor_shape)