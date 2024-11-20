def __init__(self, folder_name, transform=None, seq_len=17, only_end_points
    =False, scale_type=None, eval_mode=False):
    self.folder_name = folder_name
    self.transform = transform
    self.seq_len = seq_len
    self.only_end_points = only_end_points
    self.scale_type = scale_type
    self.eval_mode = eval_mode
    if scale_type is not None:
        assert scale_type in ['oflow', 'cr']
