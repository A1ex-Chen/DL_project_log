def get_batch_size(self) ->int:
    """Get the batch size to use for calibration."""
    return self.batch or 1
