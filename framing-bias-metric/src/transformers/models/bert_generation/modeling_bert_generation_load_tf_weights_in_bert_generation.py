def load_tf_weights_in_bert_generation(model, tf_hub_path, model_class,
    is_encoder_named_decoder=False, is_encoder=False):
    try:
        import numpy as np
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub
        import tensorflow_text
        tf.disable_eager_execution()
    except ImportError:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    tf_model = hub.Module(tf_hub_path)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        all_variables = tf_model.variable_map
        keep_track_variables = all_variables.copy()
        for key in list(all_variables.keys()):
            if 'global' in key:
                logger.info(f'Skipping {key}...')
                continue
            if not is_encoder:
                model_pointer = getattr(model, model_class)
            else:
                model_pointer = model
            is_embedding = False
            logger.info(f'Trying to match {key}...')
            sub_layers = key.split('/')[2:]
            if is_encoder_named_decoder and sub_layers[0] == 'encoder':
                logger.info(f'Skipping encoder layer {key} for decoder')
                continue
            if is_encoder and sub_layers[0] == 'decoder':
                logger.info(f'Skipping decoder layer {key} for encoder')
                continue
            for i, sub_layer in enumerate(sub_layers):
                if sub_layer == 'embeddings':
                    is_embedding = True
                elif sub_layer == 'LayerNorm':
                    is_embedding = False
                if 'layer' in sub_layer:
                    model_pointer = model_pointer.layer[int(sub_layer.split
                        ('_')[-1])]
                elif sub_layer in ['kernel', 'gamma']:
                    model_pointer = model_pointer.weight
                elif sub_layer == 'beta':
                    model_pointer = model_pointer.bias
                elif sub_layer == 'encdec':
                    model_pointer = model_pointer.crossattention.self
                elif sub_layer == 'encdec_output':
                    model_pointer = model_pointer.crossattention.output
                elif is_encoder_named_decoder and sub_layer == 'decoder':
                    model_pointer = model_pointer.encoder
                else:
                    if sub_layer == 'attention' and 'encdec' in sub_layers[
                        i + 1]:
                        continue
                    try:
                        model_pointer = getattr(model_pointer, sub_layer)
                    except AttributeError:
                        logger.info(
                            f'Skipping to initialize {key} at {sub_layer}...')
                        raise AttributeError
            array = np.asarray(sess.run(all_variables[key]))
            if not is_embedding:
                logger.info('Transposing numpy weight of shape {} for {}'.
                    format(array.shape, key))
                array = np.transpose(array)
            else:
                model_pointer = model_pointer.weight
            try:
                assert model_pointer.shape == array.shape, f'Pointer shape {model_pointer.shape} and array shape {array.shape} mismatched'
            except AssertionError as e:
                e.args += model_pointer.shape, array.shape
                raise
            logger.info(f'Initialize PyTorch weight {key}')
            model_pointer.data = torch.from_numpy(array.astype(np.float32))
            keep_track_variables.pop(key, None)
        logger.info('Weights not copied to PyTorch model: {}'.format(', '.
            join(keep_track_variables.keys())))
        return model
