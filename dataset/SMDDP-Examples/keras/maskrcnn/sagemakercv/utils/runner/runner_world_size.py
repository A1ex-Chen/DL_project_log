@property
def world_size(self):
    """
        Number of processes participating in the job.
        (distributed training)
        """
    return self._world_size
