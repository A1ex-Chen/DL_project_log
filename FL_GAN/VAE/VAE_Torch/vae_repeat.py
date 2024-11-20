def repeat(tensor, K=50):
    shape = (K,) + tuple(tensor.shape)
    return torch.cat(K * [tensor]).reshape(shape)
