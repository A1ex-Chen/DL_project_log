def check_weights(params):
    """ Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    """
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)
