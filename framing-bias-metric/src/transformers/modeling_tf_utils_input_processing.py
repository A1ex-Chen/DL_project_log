def input_processing(func, input_ids, **kwargs):
    signature = dict(inspect.signature(func).parameters)
    signature.pop('kwargs', None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = tf.Tensor, bool, int, ModelOutput, tuple, list, dict
    if 'inputs' in kwargs['kwargs_call']:
        warnings.warn(
            'The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.'
            , FutureWarning)
        output['input_ids'] = kwargs['kwargs_call'].pop('inputs')
    if 'decoder_cached_states' in kwargs['kwargs_call']:
        warnings.warn(
            'The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.'
            , FutureWarning)
        output['past_key_values'] = kwargs['kwargs_call'].pop(
            'decoder_cached_states')
    if len(kwargs['kwargs_call']) > 0:
        raise ValueError(
            f"The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}."
            )
    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(
                f'Data of type {type(v)} is not allowed only tf.Tensor is accepted for {k}.'
                )
    if isinstance(input_ids, (tuple, list)):
        for i, input in enumerate(input_ids):
            if type(input) == tf.Tensor:
                tensor_name = input.name.split(':')[0]
                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    raise ValueError(
                        f'The tensor named {input.name} does not belong to the authorized list of names {parameter_names}.'
                        )
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f'Data of type {type(input)} is not allowed only tf.Tensor is accepted for {parameter_names[i]}.'
                    )
    elif isinstance(input_ids, (dict, BatchEncoding)):
        if 'inputs' in input_ids:
            warnings.warn(
                'The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.'
                , FutureWarning)
            output['input_ids'] = input_ids.pop('inputs')
        if 'decoder_cached_states' in input_ids:
            warnings.warn(
                'The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.'
                , FutureWarning)
            output['past_key_values'] = input_ids.pop('decoder_cached_states')
        for k, v in dict(input_ids).items():
            if not isinstance(v, allowed_types):
                raise ValueError(
                    f'Data of type {type(v)} is not allowed only tf.Tensor is accepted for {k}.'
                    )
            else:
                output[k] = v
    elif isinstance(input_ids, tf.Tensor) or input_ids is None:
        output[parameter_names[0]] = input_ids
    else:
        raise ValueError(
            f'Data of type {type(input_ids)} is not allowed only tf.Tensor is accepted for {parameter_names[0]}.'
            )
    for name in parameter_names:
        if name not in list(output.keys()) and name != 'args':
            output[name] = kwargs.pop(name, signature[name].default)
    if 'args' in output:
        if output['args'] is not None and type(output['args']) == tf.Tensor:
            tensor_name = output['args'].name.split(':')[0]
            output[tensor_name] = output['args']
        else:
            output['input_ids'] = output['args']
        del output['args']
    if 'kwargs' in output:
        del output['kwargs']
    return output
