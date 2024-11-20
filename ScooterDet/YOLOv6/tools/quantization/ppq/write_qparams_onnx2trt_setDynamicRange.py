def setDynamicRange(network, json_file):
    """Sets ranges for network layers."""
    quant_param_json = json_load(json_file)
    act_quant = quant_param_json['act_quant_info']
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            print(input_tensor.name)
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = -abs(value)
            input_tensor.dynamic_range = tensor_min, tensor_max
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if act_quant.__contains__(tensor.name):
                print('\x1b[1;32mWrite quantization parameters:%s\x1b[0m' %
                    tensor.name)
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = tensor_min, tensor_max
            else:
                print(
                    '\x1b[1;31mNo quantization parameters are written: %s\x1b[0m'
                     % tensor.name)
