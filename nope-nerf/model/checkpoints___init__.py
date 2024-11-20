def __init__(self, checkpoint_dir='./chkpts', **kwargs):
    self.module_dict = kwargs
    self.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
