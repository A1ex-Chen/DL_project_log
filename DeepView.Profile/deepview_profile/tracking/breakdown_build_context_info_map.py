def build_context_info_map(self):
    """
        Builds aggregate memory/run time usage information for each tracked
        line of code.
        """
    stack = [(self, 0)]
    while len(stack) > 0:
        node, visit_count = stack.pop()
        if visit_count > 0:
            for child in node.children.values():
                context_info = ContextInfo(size_bytes=child.size_bytes,
                    run_time_ms=child.forward_ms if child.backward_ms is
                    None else child.forward_ms + child.backward_ms)
                for context in child._contexts:
                    if context in node._context_info_map:
                        node._context_info_map[context] += context_info
                    else:
                        node._context_info_map[context] = context_info
                ContextInfo.merge_map(node._context_info_map, child.
                    _context_info_map)
        else:
            node._context_info_map = {}
            stack.append((node, 1))
            for child in node.children.values():
                stack.append((child, 0))
