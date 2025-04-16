def get_transforms(cfg):
    """ Returns transforms.

    Args:
        cfg (yaml): yaml config
    """
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']
    transf_pt = data.SubsamplePoints(n_pt)
    transf_pt_val = data.SubsamplePointsSeq(n_pt_eval, random=False)
    return transf_pt, transf_pt_val
