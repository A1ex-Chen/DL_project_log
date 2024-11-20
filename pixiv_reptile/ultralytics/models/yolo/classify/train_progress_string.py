def progress_string(self):
    """Returns a formatted string showing training progress."""
    return ('\n' + '%11s' * (4 + len(self.loss_names))) % ('Epoch',
        'GPU_mem', *self.loss_names, 'Instances', 'Size')
