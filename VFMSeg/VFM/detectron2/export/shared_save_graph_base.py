def save_graph_base(net, file_name, graph_name='net', op_only=True,
    blob_rename_func=None):
    graph = None
    ops = net.op
    if blob_rename_func is not None:
        ops = _modify_blob_names(ops, blob_rename_func)
    if not op_only:
        graph = net_drawer.GetPydotGraph(ops, graph_name, rankdir='TB')
    else:
        graph = net_drawer.GetPydotGraphMinimal(ops, graph_name, rankdir=
            'TB', minimal_dependency=True)
    try:
        par_dir = os.path.dirname(file_name)
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)
        format = os.path.splitext(os.path.basename(file_name))[-1]
        if format == '.png':
            graph.write_png(file_name)
        elif format == '.pdf':
            graph.write_pdf(file_name)
        elif format == '.svg':
            graph.write_svg(file_name)
        else:
            print('Incorrect format {}'.format(format))
    except Exception as e:
        print('Error when writing graph to image {}'.format(e))
    return graph
