def save_graph(net, file_name, graph_name='net', op_only=True, blob_sizes=
    None, blob_ranges=None):
    blob_rename_f = functools.partial(_rename_blob, blob_sizes=blob_sizes,
        blob_ranges=blob_ranges)
    return save_graph_base(net, file_name, graph_name, op_only, blob_rename_f)
