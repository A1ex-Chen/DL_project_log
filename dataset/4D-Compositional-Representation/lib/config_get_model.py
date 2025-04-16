def get_model(cfg, device=None, dataset=None):
    """ Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, device=device,
        dataset=dataset)
    return model
