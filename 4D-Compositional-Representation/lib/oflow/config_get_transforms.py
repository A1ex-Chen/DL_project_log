def get_transforms(cfg):
    """ Returns transform objects.

    Args:
        cfg (yaml config): yaml config object
    """
    n_pcl = cfg['data']['n_training_pcl_points']
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']
    transf_pt = data.SubsamplePoints(n_pt)
    transf_pt_val = data.SubsamplePointsSeq(n_pt_eval, random=False)
    transf_pcl_val = data.SubsamplePointcloudSeq(n_pt_eval, random=False)
    transf_pcl = data.SubsamplePointcloudSeq(n_pcl, connected_samples=True)
    return transf_pt, transf_pt_val, transf_pcl, transf_pcl_val
