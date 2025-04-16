def add_conv_layer(model, layer_params, input_dim=None, locally_connected=False
    ):
    if len(layer_params) == 3:
        filters = layer_params[0]
        filter_len = layer_params[1]
        stride = layer_params[2]
        if locally_connected:
            if input_dim:
                model.add(LocallyConnected1D(filters, filter_len, strides=
                    stride, input_shape=(input_dim, 1)))
            else:
                model.add(LocallyConnected1D(filters, filter_len, strides=
                    stride))
        elif input_dim:
            model.add(Conv1D(filters, filter_len, strides=stride,
                input_shape=(input_dim, 1)))
        else:
            model.add(Conv1D(filters, filter_len, strides=stride))
    elif len(layer_params) == 5:
        filters = layer_params[0]
        filter_len = layer_params[1], layer_params[2]
        stride = layer_params[3], layer_params[4]
        if locally_connected:
            if input_dim:
                model.add(LocallyConnected2D(filters, filter_len, strides=
                    stride, input_shape=(input_dim, 1)))
            else:
                model.add(LocallyConnected2D(filters, filter_len, strides=
                    stride))
        elif input_dim:
            model.add(Conv2D(filters, filter_len, strides=stride,
                input_shape=(input_dim, 1)))
        else:
            model.add(Conv2D(filters, filter_len, strides=stride))
    return model
