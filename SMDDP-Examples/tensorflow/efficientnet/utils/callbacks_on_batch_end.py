def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
        now = time.time()
        elapsed_time = now - self.start_time
        steps_per_second = steps_since_last_log / elapsed_time
        examples_per_second = steps_per_second * self.batch_size
        self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
        elapsed_time_str = '{:.2f} seconds'.format(elapsed_time)
        self.logger.log(step='PARAMETER', data={'TimeHistory':
            elapsed_time_str, 'examples/second': examples_per_second,
            'steps': (self.last_log_step, self.global_steps)})
        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.scalar('global_step/sec', steps_per_second, self
                    .global_steps)
                tf.summary.scalar('examples/sec', examples_per_second, self
                    .global_steps)
        self.last_log_step = self.global_steps
        self.start_time = None
        self.throughput.append(examples_per_second)
