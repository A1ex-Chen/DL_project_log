def load_tf2_weights_in_pytorch_model(pt_model, tf_weights,
    allow_missing_keys=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
            )
        raise
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
    missing_keys_pt = []
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[
                pt_weight.data_ptr()]
            continue
        if pt_weight_name not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue
            raise AttributeError('{} not found in TF 2.0 model'.format(
                pt_weight_name))
        array, transpose = tf_weights_map[pt_weight_name]
        if transpose:
            array = numpy.transpose(array)
        if len(pt_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)
        if list(pt_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, pt_weight.shape)
            except AssertionError as e:
                e.args += pt_weight.shape, array.shape
                raise e
        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += pt_weight.shape, array.shape
            raise e
        new_pt_params_dict[pt_weight_name] = torch.from_numpy(array)
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = torch.from_numpy(
            array)
        all_tf_weights.discard(pt_weight_name)
    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict
        , strict=False)
    missing_keys += missing_keys_pt
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some weights of the TF 2.0 model were not used when initializing the PyTorch model {pt_model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).
- This IS NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect to be exactly identical (e.g. initializing a BertForSequenceClassification model from a TFBertForSequenceClassification model)."""
            )
    else:
        logger.warning(
            f"""All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.
"""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    else:
        logger.warning(
            f"""All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.
If your task is similar to the task the model of the checkpoint was trained on, you can already use {pt_model.__class__.__name__} for predictions without further training."""
            )
    logger.info('Weights or buffers not loaded from TF 2.0 model: {}'.
        format(all_tf_weights))
    return pt_model
