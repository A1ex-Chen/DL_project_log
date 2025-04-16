def _init_summary_writer(self, args, log_dir=None):
    log_dir = log_dir or args.logging_dir
    self.tb_writer = SummaryWriter(log_dir=log_dir)
