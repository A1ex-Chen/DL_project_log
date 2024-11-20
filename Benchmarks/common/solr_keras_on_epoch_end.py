def on_epoch_end(self, epoch, logs={}):
    """On every epoch end, check whether it exceeded timeout and terminate training if necessary"""
    run_end = datetime.now()
    run_duration = run_end - self.run_timestamp
    run_in_sec = run_duration.total_seconds()
    print('Current time ....%2.3f' % run_in_sec)
    if self.timeout_in_sec != -1:
        if run_in_sec >= self.timeout_in_sec:
            print('Timeout==>Runtime: %2.3fs, Maxtime: %2.3fs' % (
                run_in_sec, self.timeout_in_sec))
            self.model.stop_training = True
