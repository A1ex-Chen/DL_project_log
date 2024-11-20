def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    with self.writer.as_default():
        tf.summary.scalar(tag, value, step=step)
        self.writer.flush()
