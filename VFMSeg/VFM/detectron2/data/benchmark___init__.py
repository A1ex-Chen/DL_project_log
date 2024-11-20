def __init__(self, dataset, *, mapper, sampler=None, total_batch_size,
    num_workers=0, max_time_seconds: int=90):
    """
        Args:
            max_time_seconds (int): maximum time to spent for each benchmark
            other args: same as in `build.py:build_detection_train_loader`
        """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False, serialize=True)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    self.dataset = dataset
    self.mapper = mapper
    self.sampler = sampler
    self.total_batch_size = total_batch_size
    self.num_workers = num_workers
    self.per_gpu_batch_size = self.total_batch_size // comm.get_world_size()
    self.max_time_seconds = max_time_seconds
