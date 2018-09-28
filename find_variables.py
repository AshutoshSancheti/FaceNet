import tensorflow as tf


saver = tf.train.import_meta_graph('mobilenet_v2_checkpoints/my_model-220.meta')
imported_graph = tf.get_default_graph()
graph_op = imported_graph.get_operations()
with open('output.txt', 'w') as f:
    for i in graph_op:
        f.write(str(i))
