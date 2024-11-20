def __init__(self, print_interval, backends, start_epoch=-1, verbose=False):
    self.epoch = start_epoch
    self.iteration = -1
    self.val_iteration = -1
    self.calib_iteration = -1
    self.metrics = OrderedDict()
    self.backends = backends
    self.print_interval = print_interval
    self.verbose = verbose
    dllogger.init(backends)
