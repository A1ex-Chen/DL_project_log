def closure(x):
    l = len(x)
    s = x[:int(l / 2)]
    t = x[int(l / 2):]
    s = torch.from_numpy(s).to(dtype=dtype).to(device)
    t = torch.from_numpy(t).to(dtype=dtype).to(device)
    transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
    dists = inter_distances(transformed_arrays)
    sqrt_dist = torch.sqrt(torch.mean(dists ** 2))
    if 'mean' == reduction:
        pred = torch.mean(transformed_arrays, dim=0)
    elif 'median' == reduction:
        pred = torch.median(transformed_arrays, dim=0).values
    else:
        raise ValueError
    near_err = torch.sqrt((0 - torch.min(pred)) ** 2)
    far_err = torch.sqrt((1 - torch.max(pred)) ** 2)
    err = sqrt_dist + (near_err + far_err) * regularizer_strength
    err = err.detach().cpu().numpy().astype(np_dtype)
    return err
