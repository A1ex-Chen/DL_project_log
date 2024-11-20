def process_tensor(tensor_list, last_dim, output_mask=False):
    tensor_len = [d.shape[0] for d in tensor_list]
    tensor_max_lenth = max(tensor_len)
    d_type = tensor_list[0].dtype
    if last_dim > 0:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth, last_dim),
            dtype=d_type)
    else:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=d_type
            )
    mask_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=np.float32)
    for i, (d, l) in enumerate(zip(tensor_list, tensor_len)):
        if l > 0:
            tensor_np[i, :l] = d
            mask_np[i, :l] = 1
    if output_mask:
        return torch.from_numpy(tensor_np), torch.from_numpy(mask_np)
    else:
        return torch.from_numpy(tensor_np)
