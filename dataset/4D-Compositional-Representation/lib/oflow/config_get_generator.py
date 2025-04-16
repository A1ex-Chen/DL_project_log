def get_generator(model, cfg, device, **kwargs):
    """ Returns an OFlow generator instance.

    It provides methods to extract the final meshes from the OFlow
    representation.

    Args:
        model (nn.Module): OFlow model
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    """
    generator = generation.Generator3D(model, device=device, threshold=cfg[
        'test']['threshold'], resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'], padding=cfg
        ['generation']['padding'], sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'], n_time_steps=
        cfg['generation']['n_time_steps'], mesh_color=cfg['generation'][
        'mesh_color'], only_end_time_points=cfg['generation'][
        'only_end_time_points'], interpolate=cfg['generation'][
        'interpolate'], fix_z=cfg['generation']['fix_z'], fix_zt=cfg[
        'generation']['fix_zt'])
    return generator
