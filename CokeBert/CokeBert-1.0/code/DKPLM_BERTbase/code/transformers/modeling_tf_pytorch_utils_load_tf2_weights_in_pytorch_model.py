def load_tf2_weights_in_pytorch_model(pt_model, tf_weights,
    allow_missing_keys=False):
    """ Load TF2.0 symbolic weights in a PyTorch model
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError as e:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise e
    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())
    start_prefix_to_remove = ''
    if not any(s.startswith(pt_model.base_model_prefix) for s in
        current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + '.'
    tf_weights_map = {}
    for tf_weight in tf_weights:
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(tf_weight
            .name, start_prefix_to_remove=start_prefix_to_remove)
        tf_weights_map[pt_name] = tf_weight.numpy(), transpose
    all_tf_weights = set(list(tf_weights_map.keys()))
    loaded_pt_weights_data_ptr = {}
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[
                pt_weight.data_ptr()]
            continue
        if pt_weight_name not in tf_weights_map:
            raise ValueError('{} not found in TF 2.0 model'.format(
                pt_weight_name))
        array, transpose = tf_weights_map[pt_weight_name]
        if transpose:
            array = numpy.transpose(array)
        if len(pt_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)
        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += pt_weight.shape, array.shape
            raise e
        logger.info('Initialize PyTorch weight {}'.format(pt_weight_name))
        new_pt_params_dict[pt_weight_name] = torch.from_numpy(array)
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = torch.from_numpy(
            array)
        all_tf_weights.discard(pt_weight_name)
    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict
        , strict=False)
    if len(missing_keys) > 0:
        logger.info('Weights of {} not initialized from TF 2.0 model: {}'.
            format(pt_model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info('Weights from TF 2.0 model not used in {}: {}'.format(
            pt_model.__class__.__name__, unexpected_keys))
    logger.info('Weights or buffers not loaded from TF 2.0 model: {}'.
        format(all_tf_weights))
    return pt_model
