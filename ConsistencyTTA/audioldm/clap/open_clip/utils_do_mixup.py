def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0, -1) * mixup_lambda + torch.flip(x, dims=[0]).
        transpose(0, -1) * (1 - mixup_lambda)).transpose(0, -1)
    return out
