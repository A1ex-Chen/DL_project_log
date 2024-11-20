def progress_string(self):
    """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
    return ('\n' + '%11s' * (4 + len(self.loss_names))) % ('Epoch',
        'GPU_mem', *self.loss_names, 'Instances', 'Size')
