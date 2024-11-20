def __init__(self, renderer, cfg, use_learnt_poses=True, use_learnt_focal=
    True, device=None, render_type=None):
    self.points_batch_size = 100000
    self.renderer = renderer
    self.resolution = cfg['extract_images']['resolution']
    self.device = device
    self.use_learnt_poses = use_learnt_poses
    self.use_learnt_focal = use_learnt_focal
    self.render_type = render_type
