def _infer_graph_precision(onnx_graph: onnx.GraphProto) ->Optional[Precision]:
    import networkx as nx
    nx_graph = nx.DiGraph()

    def _get_dtype(vi):
        t = vi.type
        if hasattr(t, 'tensor_type'):
            type_id = t.tensor_type.elem_type
        else:
            raise NotImplementedError('Not implemented yet')
        return TENSOR_TYPE_TO_NP_TYPE[type_id]
    node_output2type = {vi.name: _get_dtype(vi) for vi in onnx_graph.value_info
        }
    node_outputs2node = {output_name: node for node in onnx_graph.node for
        output_name in node.output}
    node_inputs2node = {input_name: node for node in onnx_graph.node for
        input_name in node.input}
    for node in onnx_graph.node:
        node_dtype = node_output2type.get('+'.join(node.output), None)
        nx_graph.add_node(node.name, op=node.op_type, attr={a.name: a for a in
            node.attribute}, dtype=node_dtype)
        for input_name in node.input:
            prev_node = node_outputs2node.get(input_name, None)
            if prev_node:
                nx_graph.add_edge(prev_node.name, node.name)
    for input_node in onnx_graph.input:
        input_name = input_node.name
        nx_graph.add_node(input_name, op='input', dtype=_get_dtype(input_node))
        next_node = node_inputs2node.get(input_name, None)
        if next_node:
            nx_graph.add_edge(input_name, next_node.name)
    for output in onnx_graph.output:
        output_name = output.name
        nx_graph.add_node(output_name, op='output', dtype=_get_dtype(output))
        prev_node = node_outputs2node.get(output_name, None)
        if prev_node:
            nx_graph.add_edge(prev_node.name, output_name)
        else:
            LOGGER.warning(f'Could not find previous node for {output_name}')
    input_names = [n.name for n in onnx_graph.input]
    output_names = [n.name for n in onnx_graph.output]
    most_common_dtype = infer_precision(nx_graph, input_names, output_names,
        lambda node: node.get('dtype', None))
    if most_common_dtype is not None:
        precision = {np.dtype('float32'): Precision.FP32, np.dtype(
            'float16'): Precision.FP16}[most_common_dtype]
    else:
        precision = None
    return precision
