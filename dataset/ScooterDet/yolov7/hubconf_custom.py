def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')
        ) if isinstance(path_or_model, str) else path_or_model
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']
    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())
    hub_model.names = model.names
    if autoshape:
        hub_model = hub_model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    return hub_model.to(device)
