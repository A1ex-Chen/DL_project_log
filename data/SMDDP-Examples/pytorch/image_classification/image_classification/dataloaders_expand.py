def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.
        device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e
