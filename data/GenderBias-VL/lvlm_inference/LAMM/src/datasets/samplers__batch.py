def _batch(self, batch):
    """extracts samples only pertaining to this worker's batch"""
    start = self.rank * self.batch_size // self.world_size
    end = (self.rank + 1) * self.batch_size // self.world_size
    return batch[start:end]
