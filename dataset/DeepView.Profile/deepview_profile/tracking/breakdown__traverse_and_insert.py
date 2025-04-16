def _traverse_and_insert(self, root, leaf_name, stack_context):
    """
        A generator that, given a list of relevant stack frames, traverses (and
        inserts entries, if needed) the hierarchical breakdown tree, yielding
        each node along its path.
        """
    parent = root
    node_constructor = type(root)
    stack_frames = stack_context.frames
    for idx, frame in enumerate(reversed(stack_frames)):
        is_last_frame = idx == len(stack_frames) - 1
        context = frame.file_path, frame.line_number
        if context not in parent.children:
            name = leaf_name if is_last_frame else self._module_names_by_id[
                stack_frames[-idx - 2].module_id]
            if name == '':
                name = 'Model'
            new_entry = node_constructor(name, frame.module_id)
            new_entry.add_context(context)
            parent.children[context] = new_entry
        yield parent
        parent = parent.children[context]
    parent.add_name(leaf_name)
    yield parent
