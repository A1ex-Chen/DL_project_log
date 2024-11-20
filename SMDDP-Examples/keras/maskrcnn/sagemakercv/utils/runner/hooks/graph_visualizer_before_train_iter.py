@master_only
def before_train_iter(self, runner):
    if runner.iter == self.step:
        tf.summary.trace_on(graph=True, profiler=True)
