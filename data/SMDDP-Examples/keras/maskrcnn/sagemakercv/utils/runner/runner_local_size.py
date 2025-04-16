@property
def local_size(self):
    """
        Number of processes running in the same node as this runner.
        (distributed training)
        """
    return self._local_size
