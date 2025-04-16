def __init__(self, model, optimizer, cfg, device=None, optimizer_pose=None,
    pose_param_net=None, optimizer_focal=None, focal_net=None,
    optimizer_distortion=None, distortion_net=None, **kwargs):
    """model trainer

        Args:
            model (nn.Module): model
            optimizer (optimizer):pytorch optimizer object
            cfg (dict): config argument options
            device (device): Pytorch device option. Defaults to None.
            optimizer_pose (optimizer, optional): pytorch optimizer for poses. Defaults to None.
            pose_param_net (nn.Module, optional): model with pose parameters. Defaults to None.
            optimizer_focal (optimizer, optional): pytorch optimizer for focal. Defaults to None.
            focal_net (nn.Module, optional): model with focal parameters. Defaults to None.
            optimizer_distortion (optimizer, optional): pytorch optimizer for depth distortion. Defaults to None.
            distortion_net (nn.Module, optional): model with distortion parameters. Defaults to None.
        """
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.optimizer_pose = optimizer_pose
    self.pose_param_net = pose_param_net
    self.focal_net = focal_net
    self.optimizer_focal = optimizer_focal
    self.distortion_net = distortion_net
    self.optimizer_distortion = optimizer_distortion
    self.n_training_points = cfg['n_training_points']
    self.rendering_technique = cfg['type']
    self.vis_geo = cfg['vis_geo']
    self.detach_gt_depth = cfg['detach_gt_depth']
    self.pc_ratio = cfg['pc_ratio']
    self.match_method = cfg['match_method']
    self.shift_first = cfg['shift_first']
    self.detach_ref_img = cfg['detach_ref_img']
    self.scale_pcs = cfg['scale_pcs']
    self.detach_rgbs_scale = cfg['detach_rgbs_scale']
    self.vis_reprojection_every = cfg['vis_reprojection_every']
    self.nearest_limit = cfg['nearest_limit']
    self.annealing_epochs = cfg['annealing_epochs']
    self.pc_weight = cfg['pc_weight']
    self.rgb_s_weight = cfg['rgb_s_weight']
    self.rgb_weight = cfg['rgb_weight']
    self.depth_weight = cfg['depth_weight']
    self.weight_dist_2nd_loss = cfg['weight_dist_2nd_loss']
    self.weight_dist_1st_loss = cfg['weight_dist_1st_loss']
    self.depth_consistency_weight = cfg['depth_consistency_weight']
    self.loss = Loss(cfg)
