def every_n_iters(self, runner, n):
    return (runner.iter + 1) % n == 0 if n > 0 else False
