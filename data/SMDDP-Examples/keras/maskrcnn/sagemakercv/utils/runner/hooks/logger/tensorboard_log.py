@master_only
def log(self, runner):
    matched_tensors = []
    for expression in self.re_match:
        matched_tensors.extend(list(filter(re.compile(expression).match,
            runner.log_buffer.output.keys())))
    writer = tf.summary.create_file_writer(runner.tensorboard_dir)
    with writer.as_default():
        for var in matched_tensors:
            tag = '{}/{}'.format(self.name, var)
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                tf.summary.text(tag, record, step=runner.iter)
            else:
                tf.summary.scalar(tag, record, step=runner.iter)
    writer.close()
