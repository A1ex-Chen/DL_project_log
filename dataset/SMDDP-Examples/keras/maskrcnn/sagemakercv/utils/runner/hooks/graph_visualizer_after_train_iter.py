@master_only
def after_train_iter(self, runner):
    if runner.iter == self.step:
        writer = tf.summary.create_file_writer(runner.tensorboard_dir)
        with writer.as_default():
            tf.summary.trace_export(name='graph_trace', step=runner.iter,
                profiler_outdir=runner.work_dir)
        writer.close()
