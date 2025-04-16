def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if dist.is_available() and dist.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = dist.ReduceOp.SUM
        elif op == 'min':
            dop = dist.ReduceOp.MIN
        elif op == 'max':
            dop = dist.ReduceOp.MAX
        elif op == 'product':
            dop = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')
        device = torch.device('cuda')
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret
