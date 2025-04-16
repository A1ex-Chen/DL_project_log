def all_gather_dict(data):
    """
    Run all_gather on data which is a dictionary of Tensors
    """
    assert isinstance(data, dict)
    gathered_dict = {}
    for item_key in data:
        if isinstance(data[item_key], torch.Tensor):
            if is_distributed():
                data[item_key] = data[item_key].contiguous()
                tensor_list = [torch.empty_like(data[item_key]) for _ in
                    range(get_world_size())]
                dist.all_gather(tensor_list, data[item_key])
                gathered_tensor = torch.cat(tensor_list, dim=0)
            else:
                gathered_tensor = data[item_key]
            gathered_dict[item_key] = gathered_tensor
    return gathered_dict
