def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
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
    _layer_map = {'k': 'k_head', 'q': 'q_head', 'v': 'v_head', 'o':
        'post_proj', 'layer_1': 'linear_1', 'layer_2': 'linear_2',
        'rel_attn': 'attention', 'ff': 'ffn', 'kernel': 'weight', 'gamma':
        'weight', 'beta': 'bias', 'lookup_table': 'weight',
        'word_embedding': 'word_embeddings', 'input': 'embeddings'}
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer',
            'AdamWeightDecayOptimizer_1', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        if name[0] == 'generator':
            continue
        pointer = model
        skipped = False
        for m_name in name[1:]:
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(
                'layer_\\d+', m_name):
                layer_index = int(re.search('layer_(\\d+)', m_name).groups()[0]
                    )
                if layer_index < config.num_hidden_layers:
                    block_idx = 0
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    layer_index -= config.num_hidden_layers
                    pointer = pointer.layers[layer_index]
            elif m_name == 'r' and isinstance(pointer,
                FunnelRelMultiheadAttention):
                pointer = pointer.r_kernel
                break
            elif m_name in _layer_map:
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    print('Skipping {}'.format('/'.join(name)), array.shape)
                    skipped = True
                    break
        if not skipped:
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            if m_name == 'kernel':
                array = np.transpose(array)
            pointer.data = torch.from_numpy(array)
    return model
