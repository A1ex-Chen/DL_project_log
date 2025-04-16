def callback_inputs_subset(pipe, i, t, callback_kwargs):
    for tensor_name, tensor_value in callback_kwargs.items():
        assert tensor_name in pipe._callback_tensor_inputs
    return callback_kwargs
