def print_tree_deps_of(module, all_edges=None):
    """
    Prints the tree of modules depending on a given module.

    Args:
        module (`str`): The module that will be the root of the subtree we want.
        all_eges (`List[Tuple[str, str]]`, *optional*):
            The list of all edges of the tree. Will be set to `create_reverse_dependency_tree()` if not passed.
    """
    if all_edges is None:
        all_edges = create_reverse_dependency_tree()
    tree = get_tree_starting_at(module, all_edges)
    lines = [(tree[0], tree[0])]
    for index in range(1, len(tree)):
        edges = tree[index]
        start_edges = {edge[0] for edge in edges}
        for start in start_edges:
            end_edges = {edge[1] for edge in edges if edge[0] == start}
            pos = 0
            while lines[pos][1] != start:
                pos += 1
            lines = lines[:pos + 1] + [(' ' * (2 * index) + end, end) for
                end in end_edges] + lines[pos + 1:]
    for line in lines:
        print(line[0])
