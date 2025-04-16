def before_run(self, runner):
    super(TextLoggerHook, self).before_run(runner)
    self.start_iter = runner.iter
    self.json_log_path = osp.join(runner.work_dir, '{}.log.json'.format(
        runner.timestamp))
