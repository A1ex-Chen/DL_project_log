def maxpool(x, dim=-1, keepdim=False):
    """ Performs a maximum pooling operation.

    Args:
        x (tensor): input tensor
        dim (int): dimension of which the pooling operation is performed
        keepdim (bool): whether to keep the dimension
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out
