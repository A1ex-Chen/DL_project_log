def _populate_backward_data(self, node):
    self._calculate_backward_cpu_span(node)
    for ch in node.children:
        self._populate_backward_data(ch)
