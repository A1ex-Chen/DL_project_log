def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array
    for txt_name in names:
        name = txt_name.split('/')
        if any(n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer',
            'AdamWeightDecayOptimizer_1', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if '_slot_' in name[-1]:
            logger.info('Skipping {}'.format('/'.join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ['kernel', 'scale', 'embedding']:
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'self_attention':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[0]
            elif scope_names[0] == 'enc_dec_attention':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[1]
            elif scope_names[0] == 'dense_relu_dense':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[2]
            elif scope_names[0] == 'rms_norm':
                if hasattr(pointer, 'layer_norm'):
                    pointer = getattr(pointer, 'layer_norm')
                elif hasattr(pointer, 'final_layer_norm'):
                    pointer = getattr(pointer, 'final_layer_norm')
            elif scope_names[0] == 'scale':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            elif scope_names[0] == 'decoder' and name[1] == 'logits':
                continue
            elif scope_names[0] == 'logits':
                pointer = getattr(pointer, 'lm_head')
            elif scope_names[0] == 'wi' and len(scope_names
                ) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f'wi_{scope_names[1]}')
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ['kernel', 'scale', 'embedding']:
            pointer = getattr(pointer, 'weight')
        if scope_names[0] != 'embedding':
            logger.info('Transposing numpy weight of shape {} for {}'.
                format(array.shape, name))
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)
    logger.info('Weights not copied to PyTorch model: {}'.format(', '.join(
        tf_weights.keys())))
    return model
