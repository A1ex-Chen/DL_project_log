def create(name, pretrained, channels, classes, autoshape):
    """Creates a specified model

    Arguments:
        name (str): name of model, i.e. 'yolov7'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    try:
        cfg = list((Path(__file__).parent / 'cfg').rglob(f'{name}.yaml'))[0]
        model = Model(cfg, channels, classes)
        if pretrained:
            fname = f'{name}.pt'
            attempt_download(fname)
            ckpt = torch.load(fname, map_location=torch.device('cpu'))
            msd = model.state_dict()
            csd = ckpt['model'].float().state_dict()
            csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}
            model.load_state_dict(csd, strict=False)
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names
            if autoshape:
                model = model.autoshape()
        device = select_device('0' if torch.cuda.is_available() else 'cpu')
        return model.to(device)
    except Exception as e:
        s = 'Cache maybe be out of date, try force_reload=True.'
        raise Exception(s) from e
