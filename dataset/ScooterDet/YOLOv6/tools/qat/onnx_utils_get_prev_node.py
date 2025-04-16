def get_prev_node(nodes, node):
    node_input_list = node.input
    prev_node_list = []
    for node_id, node in enumerate(nodes):
        for node_output in node.output:
            if node_output in node_input_list:
                prev_node_list.append(node)
    return prev_node_list
