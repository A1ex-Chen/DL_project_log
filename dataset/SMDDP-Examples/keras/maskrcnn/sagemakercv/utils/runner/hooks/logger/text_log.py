def log(self, runner):
    log_dict = OrderedDict()
    mode = runner.mode
    log_dict['mode'] = mode
    log_dict['epoch'] = runner.epoch + 1
    log_dict['iter'] = runner.inner_iter + 1
    log_dict['lr'] = runner.current_lr.numpy()
    log_dict.update(runner.log_buffer.output)
    if runner.rank == 0:
        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
