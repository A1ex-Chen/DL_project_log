def basic_weight_init(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using xavier_normal.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear]:
        nn.init.xavier_normal_(module.weight)
