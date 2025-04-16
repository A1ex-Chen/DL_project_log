def load_tf_weights_in_electra(model, config, tf_checkpoint_path,
    discriminator_or_generator='discriminator'):
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
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        original_name: str = name
        try:
            if isinstance(model, ElectraForMaskedLM):
                name = name.replace('electra/embeddings/',
                    'generator/embeddings/')
            if discriminator_or_generator == 'generator':
                name = name.replace('electra/', 'discriminator/')
                name = name.replace('generator/', 'electra/')
            name = name.replace('dense_1', 'dense_prediction')
            name = name.replace('generator_predictions/output_bias',
                'generator_lm_head/bias')
            name = name.split('/')
            if any(n in ['global_step', 'temperature'] for n in name):
                logger.info('Skipping {}'.format(original_name))
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                    scope_names = re.split('_(\\d+)', m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                    pointer = getattr(pointer, 'weight')
                elif scope_names[0] == 'output_bias' or scope_names[0
                    ] == 'beta':
                    pointer = getattr(pointer, 'bias')
                elif scope_names[0] == 'output_weights':
                    pointer = getattr(pointer, 'weight')
                elif scope_names[0] == 'squad':
                    pointer = getattr(pointer, 'classifier')
                else:
                    pointer = getattr(pointer, scope_names[0])
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name.endswith('_embeddings'):
                pointer = getattr(pointer, 'weight')
            elif m_name == 'kernel':
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
            except AssertionError as e:
                e.args += pointer.shape, array.shape
                raise
            print('Initialize PyTorch weight {}'.format(name), original_name)
            pointer.data = torch.from_numpy(array)
        except AttributeError as e:
            print('Skipping {}'.format(original_name), name, e)
            continue
    return model
