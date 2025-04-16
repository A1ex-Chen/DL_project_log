def callback_inputs_all(pipe, i, t, callback_kwargs):
    for tensor_name in pipe._callback_tensor_inputs:
        assert tensor_name in callback_kwargs
    for tensor_name, tensor_value in callback_kwargs.items():
        assert tensor_name in pipe._callback_tensor_inputs
    return callback_kwargs
