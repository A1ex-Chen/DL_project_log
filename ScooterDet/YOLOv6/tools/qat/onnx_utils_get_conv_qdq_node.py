def get_conv_qdq_node(nodes, conv_node):
    conv_input_id = conv_node.input[0]
    dequant_node = None
    quant_node = None
    for node_id, node in enumerate(nodes):
        if node.op_type == 'DequantizeLinear' and conv_input_id in node.output:
            dequant_node = node
            break
    if dequant_node is not None:
        dequant_input_id = dequant_node.input[0]
        for node_id, node in enumerate(nodes):
            if (node.op_type == 'QuantizeLinear' and dequant_input_id in
                node.output):
                quant_node = node
                break
    return dequant_node, quant_node
