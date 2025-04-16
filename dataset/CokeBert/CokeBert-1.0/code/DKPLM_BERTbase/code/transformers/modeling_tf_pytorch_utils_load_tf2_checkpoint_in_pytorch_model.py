def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path,
    tf_inputs=None, allow_missing_keys=False):
    """ Load TF 2.0 HDF5 checkpoint in a PyTorch model
        We use HDF5 to easily do transfer learning
        (see https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError as e:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise e
    import transformers
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Loading TensorFlow weights from {}'.format(tf_checkpoint_path)
        )
    tf_model_class_name = 'TF' + pt_model.__class__.__name__
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)
    tf_model.load_weights(tf_checkpoint_path, by_name=True)
    return load_tf2_model_in_pytorch_model(pt_model, tf_model,
        allow_missing_keys=allow_missing_keys)
