def profile_ram(model, args=(), kwargs=None, num_bytes=4, detailed=False):
    graph = trace(model, args, kwargs)
    nodes = get_nodes(graph)
    placer = Placer(nodes)
    nodes = placer.place(num_bytes=num_bytes)
    df = DataFrame(index=[node.name for node in nodes], columns=['weight',
        'bias', 'input_shape', 'output_shape', 'in_tensors', 'out_tensors',
        'active_blocks', 'ram', 'scope'])
    for node in nodes:
        df.weight[node.name] = node.weights
        df.bias[node.name] = node.bias
        df.input_shape[node.name] = [x.shape for x in node.inputs]
        df.output_shape[node.name] = [x.shape for x in node.outputs]
        df.in_tensors[node.name] = [x.name for x in node.inputs]
        df.out_tensors[node.name] = [x.name for x in node.outputs]
        df.active_blocks[node.name] = node.malloc_blocks
        df.ram[node.name] = node.malloc_val
        df.scope[node.name] = node.scope
    if not detailed:
        return df.ram.max() / 2 ** 20
    return df
