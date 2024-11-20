def get_generator(model, cfg, device, **kwargs):
    """ Returns a generator instance.

    Args:
        model (nn.Module): model instance
        cfg (yaml): yaml config
        device (device): Pytorch device
    """
    generator = generation.Generator3D(model, device=device, threshold=cfg[
        'test']['threshold'], resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'], padding=cfg
        ['generation']['padding'], sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'], n_time_steps=
        cfg['generation']['n_time_steps'], only_end_time_points=cfg[
        'generation']['only_end_time_points'])
    return generator
