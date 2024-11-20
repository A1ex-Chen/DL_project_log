def get_generator(model, cfg, device):
    """ Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    """
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator
