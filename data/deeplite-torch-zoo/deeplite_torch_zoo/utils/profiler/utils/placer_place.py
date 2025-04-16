def place(self, num_bytes):
    node_edges = {}
    tensor_shapes = {}
    for node in self.nodes:
        for x in node.inputs:
            if x.name in node_edges:
                node_edges[x.name] += 1
            else:
                node_edges[x.name] = 1
            tensor_shapes[x.name] = x.shape
    active_tensors = []
    for node in self.nodes:
        for x in node.outputs:
            if x.name in node_edges:
                for _ in range(node_edges[x.name]):
                    active_tensors.append(x.name)
        malloc = set(active_tensors)
        ram = sum(prod(tensor_shapes[x]) for x in malloc)
        node.malloc_blocks = malloc
        node.malloc_val = int(ram * num_bytes)
        for x in node.inputs:
            if x.name in malloc:
                active_tensors.remove(x.name)
    return self.nodes
