def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        keys = []
        values = []
        for k in sorted(input_dict.keys()):
            keys.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, 0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict
