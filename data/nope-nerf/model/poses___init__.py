def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
    """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
    super(LearnPose, self).__init__()
    self.num_cams = num_cams
    self.init_c2w = None
    if init_c2w is not None:
        self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
    self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.
        float32), requires_grad=learn_R)
    self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.
        float32), requires_grad=learn_t)
