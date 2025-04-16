def infer_precision(nx_graph: nx.Graph, input_names: List[str],
    output_names: List[str], get_node_dtype_fn: Callable):
    node_dtypes = [nx_graph.nodes[node_name].get('dtype', None) for
        node_name in nx_graph.nodes]
    node_dtypes = [dt for dt in node_dtypes if dt is None or dt.kind not in
        ['i', 'b']]
    dtypes_counter = Counter(node_dtypes)
    return dtypes_counter.most_common()[0][0]
