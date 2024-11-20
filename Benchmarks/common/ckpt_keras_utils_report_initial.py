def report_initial(self):
    """Simply report that we are ready to run"""
    self.info('Callback initialized.')
    if self.save_interval == 0:
        self.info('Checkpoint save interval == 0 ' +
            '-> checkpoints are disabled.')
        return
    if self.metadata is not None:
        self.info("metadata='%s'" % self.metadata)
    if self.save_best:
        self.info("save_best_metric='%s'" % self.save_best_metric)
    self.info('PWD: ' + os.getcwd())
    self.info('ckpt_directory: %s' % PosixPath(self.ckpt_directory).resolve())
