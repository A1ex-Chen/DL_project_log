def __init__(self, model, cfg, device=None, optimizer_pose=None,
    pose_param_net=None, focal_net=None, **kwargs):
    self.model = model
    self.device = device
    self.optimizer_pose = optimizer_pose
    self.pose_param_net = pose_param_net
    self.focal_net = focal_net
    self.n_points = cfg['n_points']
    self.rendering_technique = cfg['type']
    self.loss = Loss_Eval()
