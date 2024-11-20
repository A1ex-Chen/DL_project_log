def get_next_node(nodes, node):
    node_output_list = node.output
    next_node_list = []
    for node_id, node in enumerate(nodes):
        for node_input in node.input:
            if node_input in node_output_list:
                next_node_list.append(node)
    return next_node_list
