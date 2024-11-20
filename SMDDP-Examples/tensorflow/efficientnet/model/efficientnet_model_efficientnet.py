def efficientnet(input: List[tf.keras.layers.Input], config: dict):
    """Creates an EfficientNet graph given the model parameters.

  This function is wrapped by the `EfficientNet` class to make a tf.keras.Model.

  Args:
    image_input: the input batch of images
    config: the model config

  Returns:
    the output of efficientnet
  """
    depth_coefficient = config['depth_coefficient']
    blocks = config['blocks']
    stem_base_filters = config['stem_base_filters']
    top_base_filters = config['top_base_filters']
    activation = get_activation(config['activation'])
    dropout_rate = config['dropout_rate']
    drop_connect_rate = config['drop_connect_rate']
    num_classes = config['num_classes']
    input_channels = config['input_channels']
    rescale_input = config['rescale_input']
    data_format = tf.keras.backend.image_data_format()
    dtype = config['dtype']
    weight_decay = config['weight_decay']
    weight_init = config['weight_init']
    images = input[0]
    if len(input) > 1:
        mix_weight = input[1]
        x = images * mix_weight + images[::-1] * (1.0 - mix_weight)
    else:
        x = images
    if data_format == 'channels_first':
        x = tf.keras.layers.Permute((3, 1, 2))(x)
    if rescale_input:
        x = preprocessing.normalize_images(x, num_channels=input_channels,
            dtype=dtype, data_format=data_format)
    x = conv2d_block(x, round_filters(stem_base_filters, config), config,
        kernel_size=[3, 3], strides=[2, 2], activation=activation, name='stem')
    num_blocks_total = sum(round_repeats(block['num_repeat'],
        depth_coefficient) for block in blocks)
    block_num = 0
    for stack_idx, block in enumerate(blocks):
        assert block['num_repeat'] > 0
        block.update({'input_filters': round_filters(block['input_filters'],
            config), 'output_filters': round_filters(block['output_filters'
            ], config), 'num_repeat': round_repeats(block['num_repeat'],
            depth_coefficient)})
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        config.update({'drop_connect_rate': drop_rate})
        block_prefix = 'stack_{}/block_0/'.format(stack_idx)
        x = mb_conv_block(x, block, config, block_prefix)
        block_num += 1
        if block['num_repeat'] > 1:
            block.update({'input_filters': block['output_filters'],
                'strides': (1, 1)})
            for block_idx in range(block['num_repeat'] - 1):
                drop_rate = drop_connect_rate * float(block_num
                    ) / num_blocks_total
                config.update({'drop_connect_rate': drop_rate})
                block_prefix = 'stack_{}/block_{}/'.format(stack_idx, 
                    block_idx + 1)
                x = mb_conv_block(x, block, config, prefix=block_prefix)
                block_num += 1
    x = conv2d_block(x, round_filters(top_base_filters, config), config,
        activation=activation, name='top')
    DENSE_KERNEL_INITIALIZER['config']['mode'] = weight_init
    x = tf.keras.layers.GlobalAveragePooling2D(name='top_pool')(x)
    if dropout_rate and dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = tf.keras.layers.Dense(num_classes, kernel_initializer=
        DENSE_KERNEL_INITIALIZER, kernel_regularizer=tf.keras.regularizers.
        l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(
        weight_decay), name='logits')(x)
    x = tf.keras.layers.Activation('softmax', name='probs', dtype=tf.float32)(x
        )
    return x
