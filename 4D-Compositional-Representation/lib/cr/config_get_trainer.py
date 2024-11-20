def get_trainer(model, optimizer, cfg, device, **kwargs):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): model instance
        optimzer (torch.optim): Pytorch optimizer
        cfg (yaml): yaml config
        device (device): Pytorch device
    """
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    eval_sample = cfg['training']['eval_sample']
    trainer = training.Trainer(model, optimizer, device=device, input_type=
        input_type, threshold=threshold)
    return trainer
