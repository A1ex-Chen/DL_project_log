def wrap_frozen_graph(gd, inputs, outputs):
    x = tf.compat.v1.wrap_function(lambda : tf.compat.v1.import_graph_def(
        gd, name=''), [])
    ge = x.graph.as_graph_element
    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure
        (ge, outputs))
