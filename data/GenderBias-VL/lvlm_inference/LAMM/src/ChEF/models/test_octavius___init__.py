def __init__(self, **args):
    super().__init__()
    self.args = args
    self.conv_mode = args['conv_mode']
    self.task_type = 'normal'
    self.max_tgt_len = args['max_tgt_len']
    self.model = Octavius(**args)
    delta_ckpt = torch.load(args['delta_ckpt_path'], 'cpu')
    info = self.model.load_state_dict(delta_ckpt, strict=False)
    self.model = self.model.eval().half()
    self.move_to_device()
