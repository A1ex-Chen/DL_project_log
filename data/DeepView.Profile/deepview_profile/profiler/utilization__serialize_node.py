def _serialize_node(respNode, internalNode):
    respNode.slice_id = internalNode['slice_id']
    respNode.name = internalNode['name']
    respNode.start = int(internalNode['start'])
    respNode.end = int(internalNode['end'])
    respNode.cpu_forward = int(internalNode['cpu_forward'])
    respNode.cpu_forward_span = int(internalNode['cpu_forward_span'])
    respNode.gpu_forward = int(internalNode['gpu_forward'])
    respNode.gpu_forward_span = int(internalNode['gpu_forward_span'])
    respNode.cpu_backward = int(internalNode['cpu_backward'])
    respNode.cpu_backward_span = int(internalNode['cpu_backward_span'])
    respNode.gpu_backward = int(internalNode['gpu_backward'])
    respNode.gpu_backward_span = int(internalNode['gpu_backward_span'])
    for ch in internalNode['children']:
        addRespNode = respNode.children.add()
        _serialize_node(addRespNode, ch)
