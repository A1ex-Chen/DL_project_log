def every_n_inner_iters(self, runner, n):
    return (runner.inner_iter + 1) % n == 0 if n > 0 else False
