@staticmethod
def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.float64
        ).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret
