def every_n_epochs(self, runner, n):
    return (runner.epoch + 1) % n == 0 if n > 0 else False
