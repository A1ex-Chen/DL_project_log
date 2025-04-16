def patch_device(module):
    try:
        graphs = [module.graph] if hasattr(module, 'graph') else []
    except RuntimeError:
        graphs = []
    if hasattr(module, 'forward1'):
        graphs.append(module.forward1.graph)
    for graph in graphs:
        for node in graph.findAllNodes('prim::Constant'):
            if 'value' in node.attributeNames() and str(node['value']
                ).startswith('cuda'):
                node.copyAttributes(device_node)
