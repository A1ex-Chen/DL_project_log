def onnx_add_insert_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    patterns = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == 'Add':
            same_input_node_list = []
            same_input = None
            for add_input in node.input:
                for other_id, other_node in enumerate(nodes):
                    if other_id != node_id:
                        for other_input in other_node.input:
                            if other_input == add_input:
                                same_input_node_list.append(other_node)
                                same_input = other_input
                                break
            if len(same_input_node_list) == 1 and same_input_node_list[0
                ].op_type == 'QuantizeLinear':
                prev_add_node = search_node_by_output_id(nodes, same_input)
                dequant_node = get_next_node(nodes, same_input_node_list[0])[0]
                patterns.append((node, prev_add_node, same_input_node_list[
                    0], dequant_node, same_input))
    print(patterns)
    for pattern in patterns:
        add_node, prev_add_node, quant_node, dequant_node, same_input = pattern
        dq_x, dq_s, dq_z = dequant_node.input
        new_quant_node = onnx.helper.make_node('QuantizeLinear', inputs=
            quant_node.input, outputs=[prev_add_node.name + '_Dequant'],
            name=prev_add_node.name + '_QuantizeLinear')
        new_dequant_node = onnx.helper.make_node('DequantizeLinear', inputs
            =[prev_add_node.name + '_Dequant', dq_s, dq_z], outputs=[
            prev_add_node.name + '_Add'], name=prev_add_node.name +
            '_DequantizeLinear')
        add_node.input.remove(same_input)
        add_node.input.append(prev_add_node.name + '_Add')
        for node_id, node in enumerate(graph.node):
            if node.name == prev_add_node.name:
                graph.node.insert(node_id + 1, new_quant_node)
                graph.node.insert(node_id + 2, new_dequant_node)
    return onnx_replica
