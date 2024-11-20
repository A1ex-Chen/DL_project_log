def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=
    None, allow_missing_keys=False):
    """ Load pytorch state_dict in a TF 2.0 model.
    """
    try:
        import torch
        import tensorflow as tf
        from tensorflow.python.keras import backend as K
    except ImportError as e:
        logger.error(
            'Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise e
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)
    start_prefix_to_remove = ''
    if not any(s.startswith(tf_model.base_model_prefix) for s in
        pt_state_dict.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + '.'
    symbolic_weights = (tf_model.trainable_weights + tf_model.
        non_trainable_weights)
    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(sw_name,
            start_prefix_to_remove=start_prefix_to_remove)
        assert name in pt_state_dict, '{} not found in PyTorch model'.format(
            name)
        array = pt_state_dict[name].numpy()
        if transpose:
            array = numpy.transpose(array)
        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)
        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += symbolic_weight.shape, array.shape
            raise e
        logger.info('Initialize TF weight {}'.format(symbolic_weight.name))
        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)
    K.batch_set_value(weight_value_tuples)
    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)
    logger.info('Weights or buffers not loaded from PyTorch model: {}'.
        format(all_pytorch_weights))
    return tf_model
