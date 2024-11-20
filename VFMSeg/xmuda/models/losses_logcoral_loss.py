def logcoral_loss(x_src, x_trg):
    """
    Geodesic loss (log coral loss), reference:
    https://github.com/pmorerio/minimal-entropy-correlation-alignment/blob/master/svhn2mnist/model.py
    :param x_src: source features of size (N, ..., F), where N is the batch size and F is the feature size
    :param x_trg: target features of size (N, ..., F), where N is the batch size and F is the feature size
    :return: geodesic distance between the x_src and x_trg
    """
    assert x_src.shape[-1] == x_trg.shape[-1]
    assert x_src.dim() >= 2
    batch_size = x_src.shape[0]
    if x_src.dim() > 2:
        x_src = x_src.flatten(end_dim=-2)
        x_trg = x_trg.flatten(end_dim=-2)
    x_src = x_src - torch.mean(x_src, 0)
    x_trg = x_trg - torch.mean(x_trg, 0)
    factor = 1.0 / (batch_size - 1)
    cov_src = factor * torch.mm(x_src.t(), x_src)
    cov_trg = factor * torch.mm(x_trg.t(), x_trg)
    condition = (cov_src > 1e+30).any() or (cov_trg > 1e+30).any(
        ) or torch.isnan(cov_src).any() or torch.isnan(cov_trg).any()
    cov_src = torch.where(torch.full_like(cov_src, condition, dtype=torch.
        uint8), torch.eye(cov_src.shape[0], device=cov_src.device), cov_src)
    cov_trg = torch.where(torch.full_like(cov_trg, condition, dtype=torch.
        uint8), torch.eye(cov_trg.shape[0], device=cov_trg.device), cov_trg)
    if condition:
        logger = logging.getLogger('xmuda.train')
        logger.info(
            'Big number > 1e30 or nan in covariance matrix, return loss of 0 to prevent error in SVD decomposition.'
            )
    _, e_src, v_src = cov_src.svd()
    _, e_trg, v_trg = cov_trg.svd()
    log_cov_src = torch.mm(v_src, torch.mm(torch.diag(torch.log(e_src)),
        v_src.t()))
    log_cov_trg = torch.mm(v_trg, torch.mm(torch.diag(torch.log(e_trg)),
        v_trg.t()))
    return torch.mean((log_cov_src - log_cov_trg) ** 2)
