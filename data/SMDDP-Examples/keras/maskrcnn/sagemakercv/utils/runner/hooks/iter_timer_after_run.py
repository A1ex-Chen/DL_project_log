def after_run(self, runner):
    end_time = datetime.datetime.now()
    if runner.rank == 0:
        runner.logger.info('End time: {}'.format(str(self.start_time)))
        runner.logger.info('Elapsed time: {}'.format(str(end_time - self.
            start_time)))
