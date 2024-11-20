def scan_params(self, gParameters):
    """Simply translate gParameters into instance fields"""
    self.epoch_max = param(gParameters, 'epochs', ParamRequired(),
        ParamType.INTEGER_NN)
    self.skip_epochs = param(gParameters, 'ckpt_skip_epochs', 0, ParamType.
        INTEGER_NN)
    self.ckpt_directory = param(gParameters, 'ckpt_directory', './save',
        ParamType.STRING)
    self.save_best = param(gParameters, 'ckpt_save_best', True, ParamType.
        BOOLEAN)
    self.save_best_metric = param(gParameters, 'ckpt_save_best_metric',
        None, ParamType.STRING)
    self.best_metric_last = param(gParameters, 'ckpt_best_metric_last',
        None, ParamType.FLOAT)
    if self.best_metric_last is None:
        import math
        self.best_metric_last = math.inf
    self.save_interval = param(gParameters, 'ckpt_save_interval', 1,
        ParamType.INTEGER_NN)
    self.save_weights_only = param(gParameters, 'ckpt_save_weights_only', 
        True, ParamType.BOOLEAN)
    self.checksum_enabled = param(gParameters, 'ckpt_checksum', False,
        ParamType.BOOLEAN)
    self.keep_mode = param(gParameters, 'ckpt_keep_mode', 'linear',
        ParamType.STRING, allowed=[None, 'all', 'linear'])
    self.keep_limit = param(gParameters, 'ckpt_keep_limit', 1000000,
        ParamType.INTEGER_GZ)
    self.metadata = param(gParameters, 'metadata', None, ParamType.STRING)
    self.timestamp_last = param(gParameters, 'ckpt_timestamp_last', None,
        ParamType.STRING)
    self.cwd = os.getcwd()
