def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tf.summary.create_file_writer(log_dir)
