def _shard_size(self):
    """
        Total number of samples handled by a single GPU in a single epoch.
        """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if self.drop_last:
        divisor = world_size * self.batch_size * self.grad_accumulation_steps
        return self.dataset_size // divisor * divisor // world_size
    else:
        return int(math.ceil(self.dataset_size / world_size))
