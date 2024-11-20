def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=
    None, allow_missing_keys=False):
    """Load pytorch state_dict in a TF 2.0 model."""
    try:
        import tensorflow as tf
        import torch
        from tensorflow.python.keras import backend as K
    except ImportError:
        logger.error(
            'Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)
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
    tf_loaded_numel = 0
    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    missing_keys = []
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(sw_name,
            start_prefix_to_remove=start_prefix_to_remove)
        if name not in pt_state_dict:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                if any(re.search(pat, name) is not None for pat in tf_model
                    ._keys_to_ignore_on_load_missing):
                    continue
            raise AttributeError('{} not found in PyTorch model'.format(name))
        array = pt_state_dict[name].numpy()
        if transpose:
            array = numpy.transpose(array)
        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)
        if list(symbolic_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, symbolic_weight.shape)
            except AssertionError as e:
                e.args += symbolic_weight.shape, array.shape
                raise e
        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += symbolic_weight.shape, array.shape
            raise e
        tf_loaded_numel += array.size
        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)
    K.batch_set_value(weight_value_tuples)
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)
    logger.info('Loaded {:,} parameters in the TF 2.0 model.'.format(
        tf_loaded_numel))
    unexpected_keys = list(all_pytorch_weights)
    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is
                None]
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat,
                k) is None]
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some weights of the PyTorch model were not used when initializing the TF 2.0 model {tf_model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model)."""
            )
    else:
        logger.warning(
            f"""All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.
"""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the PyTorch model and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    else:
        logger.warning(
            f"""All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.
If your task is similar to the task the model of the checkpoint was trained on, you can already use {tf_model.__class__.__name__} for predictions without further training."""
            )
    return tf_model
