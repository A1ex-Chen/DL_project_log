def onnx_remove_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    in_rename_map = {}
    scale_node_list = []
    zero_node_list = []
    activation_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == 'QuantizeLinear':
            in_name = node.input[0]
            scale_name = node.input[1]
            zero_name = node.input[2]
            out_name = node.output[0]
            in_rename_map[out_name] = in_name
            scale_node_list.append(scale_name)
            zero_node_list.append(zero_name)
            for i, node in enumerate(graph.node):
                if node.output[0] == scale_name:
                    if len(node.attribute[0].t.dims) == 0:
                        val = np.frombuffer(node.attribute[0].t.raw_data,
                            dtype=np.float32)[0]
                        if in_name in activation_map.keys():
                            old_val = struct.unpack('!f', bytes.fromhex(
                                activation_map[in_name]))[0]
                            if val > old_val:
                                activation_map[in_name] = struct.pack('>f', val
                                    ).hex()
                        else:
                            activation_map[in_name] = struct.pack('>f', val
                                ).hex()
            graph.node.remove(nodes[node_id])
    for node_id, node in enumerate(graph.node):
        for in_id, in_name in enumerate(node.input):
            if in_name in in_rename_map.keys():
                node.input[in_id] = in_rename_map[in_name]
    in_rename_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == 'DequantizeLinear':
            in_name = node.input[0]
            scale_name = node.input[1]
            zero_name = node.input[2]
            out_name = node.output[0]
            in_rename_map[out_name] = in_name
            graph.node.remove(nodes[node_id])
            scale_node_list.append(scale_name)
            zero_node_list.append(zero_name)
    for node_id, node in enumerate(graph.node):
        for in_id, in_name in enumerate(node.input):
            if in_name in in_rename_map.keys():
                node.input[in_id] = in_rename_map[in_name]
    nodes = graph.node
    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.name == node_name:
                graph.node.remove(nodes[node_id])
    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.output[0] == node_name:
                graph.node.remove(nodes[node_id])
    return onnx_replica, activation_map
