@master_only
def after_train_iter(self, runner):
    if self.every_n_inner_iters(runner, self.record_interval):
        with runner.writer.as_default():
            for var, metric in self.system_metrics.items():
                tag = '{}/{}'.format('system', var)
                record = mean(metric[-self.rolling_mean_interval:])
                tf.summary.scalar(tag, record, step=runner.iter)
    runner.writer.flush()
