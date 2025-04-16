def list_of_scalars_summary(self, tag_value_pairs, step):
    """Log scalar variables."""
    with self.writer.as_default():
        for tag, value in tag_value_pairs:
            tf.summary.scalar(tag, value, step=step)
        self.writer.flush()
