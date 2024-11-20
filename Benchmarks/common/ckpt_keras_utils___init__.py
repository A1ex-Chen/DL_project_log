def __init__(self, gParameters, logger='DEFAULT', verbose=True):
    """
        Parameters
        ----------
            logger : Logger
                The logger to use.
                May be None to disable or "DEFAULT" to use the default.
            verbose : boolean
                If True, more verbose logging
                Passed to helper_utils.set_up_logger(verbose) for this logger
        """
    self.logger = logger
    if self.logger == 'DEFAULT':
        import logging
        self.logger = logging.getLogger('CandleCheckpointCallback')
        set_up_logger('save/ckpt.log', self.logger, verbose=verbose,
            fmt_line='%(asctime)s CandleCheckpoint: %(message)s')
    self.scan_params(gParameters)
    self.epochs = []
    self.epoch_best = 0
    self.report_initial()
