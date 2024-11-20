def get_nodes(graph):
    nodes = []
    for i, node in enumerate(graph.nodes):
        if 'aten' in node.operator:
            inputs = []
            outputs = []
            weights = []
            bias = []
            if 'conv' in node.operator:
                weights = node.inputs[1].shape
                if node.inputs[2].shape is not None:
                    bias = node.inputs[2].shape
                inputs.append(Tensor(name=node.inputs[0].name, dtype=node.
                    inputs[0].dtype, shape=node.inputs[0].shape, scope=node
                    .scope))
            elif 'mm' in node.operator:
                weights = node.inputs[2].shape
                if node.inputs[0].shape is not None:
                    bias = node.inputs[0].shape
                inputs.append(Tensor(name=node.inputs[1].name, dtype=node.
                    inputs[1].dtype, shape=node.inputs[1].shape, scope=node
                    .scope))
            elif node.operator in ['aten::batch_norm', 'aten::instance_norm']:
                if node.inputs[1].shape is not None:
                    weights = node.inputs[1].shape
                    bias = node.inputs[2].shape
                inputs.append(Tensor(name=node.inputs[0].name, dtype=node.
                    inputs[0].dtype, shape=node.inputs[0].shape, scope=node
                    .scope))
            elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
                if node.inputs[2].shape is not None:
                    weights = node.inputs[2].shape
                    bias = node.inputs[2].shape
                inputs.append(Tensor(name=node.inputs[0].name, dtype=node.
                    inputs[0].dtype, shape=node.inputs[0].shape, scope=node
                    .scope))
            else:
                for x in node.inputs:
                    if x.shape is not None:
                        if x.ndim > 1:
                            inputs.append(Tensor(name=x.name, dtype=x.dtype,
                                shape=x.shape, scope=node.scope))
            for x in node.outputs:
                outputs.append(Tensor(name=x.name, dtype=x.dtype, shape=x.
                    shape, scope=node.scope))
            nodes.append(Layer(name='{}_{}'.format(i, node.operator),
                inputs=inputs, outputs=outputs, weights=weights, bias=bias,
                scope=node.scope))
    return nodes
