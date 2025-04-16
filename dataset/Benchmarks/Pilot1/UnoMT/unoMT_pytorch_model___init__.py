def __init__(self, args, use_cuda, device):
    self.args = args
    self.use_cuda = use_cuda
    self.device = device
    self.config_data_loaders()
    self.build_data_loaders()
    self.build_nn()
    self.config_optimization()
