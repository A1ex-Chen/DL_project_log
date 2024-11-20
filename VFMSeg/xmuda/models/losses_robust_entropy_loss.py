def robust_entropy_loss(x, eta=2.0):
    """
    Robust entropy loss.
    Reference: https://github.com/YanchaoYang/FDA

    :param x: Logits before softmax, size: (batch_size, num_classes, number of points).
    :param eta: Hyperparameter of the robust entropy loss.
    :return: Scalar entropy loss.
    """
    if x.dim() != 3:
        raise ValueError(
            f'Expected 3-dimensional vector, but received {x.dim()}')
    P = F.softmax(x, dim=1)
    logP = F.log_softmax(x, dim=1)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    num_classes = x.shape[1]
    ent = ent / torch.log(torch.tensor(num_classes, dtype=torch.float))
    ent = ent ** 2.0 + 1e-08
    ent = ent ** eta
    return ent.mean()
