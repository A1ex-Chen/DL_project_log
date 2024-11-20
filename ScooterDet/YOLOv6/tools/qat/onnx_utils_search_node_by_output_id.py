def search_node_by_output_id(nodes, output_id: str):
    prev_node = None
    for node_id, node in enumerate(nodes):
        if output_id in node.output:
            prev_node = node
            break
    return prev_node
