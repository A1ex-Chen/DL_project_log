def get_trainer(model, optimizer, cfg, device):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(model, optimizer, cfg,
        device)
    return trainer
