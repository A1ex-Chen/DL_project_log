def __init__(self, model_path, device=None, task_type='normal', **kwargs):
    self.conv_mode = 'simple'
    self.model = LAMMPEFTModel(**kwargs)
    delta_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    self.model.load_state_dict(delta_ckpt, strict=False)
    self.model = self.model.eval().half()
    self.task_type = task_type
    self.move_to_device(device)
    self.model.device = device
