def load_tf_weights_in_xlnet(model, config, tf_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            'Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)
    for name, pointer in tf_to_pt_map.items():
        logger.info('Importing {}'.format(name))
        if name not in tf_weights:
            logger.info('{} not in tf pre-trained weights, skipping'.format
                (name))
            continue
        array = tf_weights[name]
        if 'kernel' in name and ('ff' in name or 'summary' in name or 
            'logit' in name):
            logger.info('Transposing')
            array = np.transpose(array)
        if isinstance(pointer, list):
            assert len(pointer) == array.shape[0
                ], f'Pointer length {len(pointer)} and array length {array.shape[0]} mismatched'
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape, f'Pointer shape {p_i.shape} and array shape {arr_i.shape} mismatched'
                except AssertionError as e:
                    e.args += p_i.shape, arr_i.shape
                    raise
                logger.info('Initialize PyTorch weight {} for layer {}'.
                    format(name, i))
                p_i.data = torch.from_numpy(arr_i)
        else:
            try:
                assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            logger.info('Initialize PyTorch weight {}'.format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    logger.info('Weights not copied to PyTorch model: {}'.format(', '.join(
        tf_weights.keys())))
    return model
