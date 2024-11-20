def get_tree_starting_at(module: str, edges: List[Tuple[str, str]]) ->List[
    Union[str, List[str]]]:
    """
    Returns the tree starting at a given module following all edges.

    Args:
        module (`str`): The module that will be the root of the subtree we want.
        eges (`List[Tuple[str, str]]`): The list of all edges of the tree.

    Returns:
        `List[Union[str, List[str]]]`: The tree to print in the following format: [module, [list of edges
        starting at module], [list of edges starting at the preceding level], ...]
    """
    vertices_seen = [module]
    new_edges = [edge for edge in edges if edge[0] == module and edge[1] !=
        module and '__init__.py' not in edge[1]]
    tree = [module]
    while len(new_edges) > 0:
        tree.append(new_edges)
        final_vertices = list({edge[1] for edge in new_edges})
        vertices_seen.extend(final_vertices)
        new_edges = [edge for edge in edges if edge[0] in final_vertices and
            edge[1] not in vertices_seen and '__init__.py' not in edge[1]]
    return tree
