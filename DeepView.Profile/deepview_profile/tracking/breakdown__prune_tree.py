def _prune_tree(self, root):
    stack = [(root, None, None)]
    while len(stack) != 0:
        node, key, parent = stack.pop()
        if len(node.children) == 1 and parent is not None:
            child = next(iter(node.children.values()))
            child.add_context(key)
            parent.children[key] = child
            node.children.clear()
            stack.append((child, key, parent))
        else:
            for key_to_child, child in node.children.items():
                stack.append((child, key_to_child, node))
