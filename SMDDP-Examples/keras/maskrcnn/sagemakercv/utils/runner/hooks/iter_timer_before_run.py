def before_run(self, runner):
    self.start_time = datetime.datetime.now()
    if runner.rank == 0:
        runner.logger.info('Start time: {}'.format(str(self.start_time)))
