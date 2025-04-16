@property
def train_batch_size_per_device(self):
    """
        Maximum training epochs.
        """
    return self.train_batch_size // self.world_size
