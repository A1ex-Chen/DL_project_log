def _convert_node_to_dict(self, node):
    newEntry = {'slice_id': node.slice_id, 'name': node.name, 'start': node
        .start, 'end': node.end, 'cpu_forward': node.duration,
        'cpu_forward_span': node.cpu_forward, 'gpu_forward': node.
        gpu_forward, 'gpu_forward_span': node.gpu_forward_span,
        'cpu_backward': node.cpu_backward, 'cpu_backward_span': node.
        cpu_backward_span, 'gpu_backward': node.gpu_backward,
        'gpu_backward_span': node.gpu_backward_span, 'children': list()}
    for ch in node.children:
        newEntry['children'].append(self._convert_node_to_dict(ch))
    return newEntry
