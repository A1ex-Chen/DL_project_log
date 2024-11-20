def __init__(self, checkpoint_dir='./chkpts', initialize_from=None,
    initialization_file_name='model_best.pt', **kwargs):
    self.module_dict = kwargs
    self.checkpoint_dir = checkpoint_dir
    self.initialize_from = initialize_from
    self.initialization_file_name = initialization_file_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
