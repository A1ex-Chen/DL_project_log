def _accumulate_backward_slices_to_node(self, node):
    for ch in node.children:
        node.cpu_backward_slices.extend(self.
            _accumulate_backward_slices_to_node(ch))
    return node.cpu_backward_slices
