@try_export
def export_pb(self, keras_model, prefix=colorstr('TensorFlow GraphDef:')):
    """YOLOv8 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    LOGGER.info(
        f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = self.file.with_suffix('.pb')
    m = tf.function(lambda x: keras_model(x))
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape,
        keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.
        parent), name=f.name, as_text=False)
    return f, None
