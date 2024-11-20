def visit_FunctionDef(self, node):
    if self.function_node is not None:
        return
    if node.name != 'deepview_input_provider':
        return
    self.function_node = node
