@staticmethod
def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64).expand(x
        .shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret
