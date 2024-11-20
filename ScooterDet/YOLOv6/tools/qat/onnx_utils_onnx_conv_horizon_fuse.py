def onnx_conv_horizon_fuse(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    pattern = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == 'Add':
            avail_count = 0
            for input_id in node.input:
                prev_node = search_node_by_output_id(graph.node, input_id)
                if prev_node is not None:
                    if prev_node.op_type in ['BatchNormalization', 'Conv'
                        ] and len(prev_node.output) == 1:
                        avail_count += 1
            if avail_count == 2:
                pattern.append(node)
    for add_node in pattern:
        prev_add_node_list = get_prev_node(nodes, add_node)
        conv_node_list = []
        for node in prev_add_node_list:
            if node.op_type == 'BatchNormalization':
                prev_node_list = get_prev_node(nodes, node)
                assert len(prev_node_list) == 1 and prev_node_list[0
                    ].op_type == 'Conv', 'Conv horizon fusion pattern not match'
                conv_node_list.append(prev_node_list[0])
            else:
                conv_node_list.append(node)
        qdq_node_list = []
        for node in conv_node_list:
            dequant_node, quant_node = get_conv_qdq_node(nodes, node)
            assert dequant_node is not None and quant_node is not None, 'Conv horizon fusion pattern not match'
            qdq_node_list.extend((dequant_node, quant_node))
        scale_node_list = []
        for qdq_node in qdq_node_list:
            scale_iput_id = qdq_node.input[1]
            for node in nodes:
                if scale_iput_id in node.output:
                    scale_node_list.append(node)
        max = 0
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=
                np.float32)[0]
            print(val)
            if max < val:
                max = val
        for scale_node in scale_node_list:
            scale_node.attribute[0].t.raw_data = bytes(struct.pack('f', max))
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=
                np.float32)[0]
            print(val)
    return onnx_replica
