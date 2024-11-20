def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""
    try:
        import torch
    except ImportError:
        logger.error(
            'Loading Flax weights in PyTorch requires both PyTorch and Flax to be installed. Please see https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions.'
            )
        raise
    is_type_bf16 = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype ==
        jnp.bfloat16, flax_state)).values()
    if any(is_type_bf16):
        logger.warning(
            'Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` before loading those in PyTorch model.'
            )
        flax_state = jax.tree_util.tree_map(lambda params: params.astype(np
            .float32) if params.dtype == jnp.bfloat16 else params, flax_state)
    pt_model.base_model_prefix = ''
    flax_state_dict = flatten_dict(flax_state, sep='.')
    pt_model_dict = pt_model.state_dict()
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())
    for flax_key_tuple, flax_tensor in flax_state_dict.items():
        flax_key_tuple_array = flax_key_tuple.split('.')
        if flax_key_tuple_array[-1] == 'kernel' and flax_tensor.ndim == 4:
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ['weight']
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))
        elif flax_key_tuple_array[-1] == 'kernel':
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ['weight']
            flax_tensor = flax_tensor.T
        elif flax_key_tuple_array[-1] == 'scale':
            flax_key_tuple_array = flax_key_tuple_array[:-1] + ['weight']
        if 'time_embedding' not in flax_key_tuple_array:
            for i, flax_key_tuple_string in enumerate(flax_key_tuple_array):
                flax_key_tuple_array[i] = flax_key_tuple_string.replace('_0',
                    '.0').replace('_1', '.1').replace('_2', '.2').replace('_3',
                    '.3').replace('_4', '.4').replace('_5', '.5').replace('_6',
                    '.6').replace('_7', '.7').replace('_8', '.8').replace('_9',
                    '.9')
        flax_key = '.'.join(flax_key_tuple_array)
        if flax_key in pt_model_dict:
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f'Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}.'
                    )
            else:
                flax_tensor = np.asarray(flax_tensor) if not isinstance(
                    flax_tensor, np.ndarray) else flax_tensor
                pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
                missing_keys.remove(flax_key)
        else:
            unexpected_keys.append(flax_key)
    pt_model.load_state_dict(pt_model_dict)
    missing_keys = list(missing_keys)
    if len(unexpected_keys) > 0:
        logger.warning(
            f"""Some weights of the Flax model were not used when initializing the PyTorch model {pt_model.__class__.__name__}: {unexpected_keys}
- This IS expected if you are initializing {pt_model.__class__.__name__} from a Flax model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).
- This IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect to be exactly identical (e.g. initializing a BertForSequenceClassification model from a FlaxBertForSequenceClassification model)."""
            )
    if len(missing_keys) > 0:
        logger.warning(
            f"""Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly initialized: {missing_keys}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
            )
    return pt_model
