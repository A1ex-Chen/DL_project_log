def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([(init_dim * np.arange(
        n_tile) + i) for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
