def profile_macs(model, args=(), kwargs=None, reduction=sum):
    results = {}
    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node] = func(node)
                break
        else:
            warnings.warn(f'No handlers found: "{node.operator}". Skipped.')
    if reduction is not None:
        return reduction(results.values())
    return results
