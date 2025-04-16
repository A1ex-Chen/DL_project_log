def end_of_epoch(self, runner):
    return runner.inner_iter + 1 == runner.num_examples
