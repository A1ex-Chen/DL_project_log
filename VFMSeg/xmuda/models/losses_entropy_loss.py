def entropy_loss(v):
    """
    Entropy loss.
    Reference: https://github.com/valeoai/ADVENT

    :param v: Input tensor after softmax of size (num_points, num_classes).
    :return: Scalar entropy loss.
    """
    if v.dim() == 2:
        v = v.transpose(0, 1)
        v = v.unsqueeze(0)
    assert v.dim() == 3
    n, c, p = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * p * np.
        log2(c))
