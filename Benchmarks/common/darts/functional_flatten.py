def flatten(tensor):
    """Flatten a tensor.

    Parameters
    ----------
    tensor : torch.tensor

    Returns
    -------
    Flattened tensor

    Example
    -------
    >>> x = torch.tensor([[0,1],[2,3]])
    >>> x_flattened = flatten(x)
    >>> print(x)
    >>> tensor([[0, 1],
                [2, 3]])
    >>> print(x_flattened)
    >>> tensor([0, 1, 2, 3])
    """
    return torch.cat([x.view(-1) for x in tensor])
