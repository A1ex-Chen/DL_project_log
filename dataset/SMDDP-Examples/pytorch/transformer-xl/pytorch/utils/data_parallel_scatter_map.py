def scatter_map(obj):
    if isinstance(obj, torch.Tensor):
        try:
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        except:
            print('obj', obj.size())
            print('dim', dim)
            print('chunk_sizes', chunk_sizes)
            quit()
    if isinstance(obj, tuple) and len(obj) > 0:
        return list(zip(*map(scatter_map, obj)))
    if isinstance(obj, list) and len(obj) > 0:
        return list(map(list, zip(*map(scatter_map, obj))))
    if isinstance(obj, dict) and len(obj) > 0:
        return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
    return [obj for targets in target_gpus]
