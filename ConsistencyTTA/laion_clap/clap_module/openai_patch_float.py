def patch_float(module):
    try:
        graphs = [module.graph] if hasattr(module, 'graph') else []
    except RuntimeError:
        graphs = []
    if hasattr(module, 'forward1'):
        graphs.append(module.forward1.graph)
    for graph in graphs:
        for node in graph.findAllNodes('aten::to'):
            inputs = list(node.inputs())
            for i in [1, 2]:
                if inputs[i].node()['value'] == 5:
                    inputs[i].node().copyAttributes(float_node)
