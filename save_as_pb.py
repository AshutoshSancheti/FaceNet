import tensorflow as tf

meta_path = 'mobilenet_v2_checkpoints/my_model-1100.meta' # Your .meta file

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)
    tf.get_default_session().run(tf.global_variables_initializer())
    tf.get_default_session().run(tf.local_variables_initializer())

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('mobilenet_v2_checkpoints'))

    # Output nodes
    output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
    #print (output_node_names)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

    # Save the frozen graph
    with open('faceNet_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
