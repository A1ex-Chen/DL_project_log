def benchmark_workers(self, num_iter, warmup=10):
    """
        Benchmark the dataloader by tuning num_workers to [0, 1, self.num_workers].
        """
    candidates = [0, 1]
    if self.num_workers not in candidates:
        candidates.append(self.num_workers)
    dataset = MapDataset(self.dataset, self.mapper)
    for n in candidates:
        loader = build_batch_data_loader(dataset, self.sampler, self.
            total_batch_size, num_workers=n)
        self._benchmark(iter(loader), num_iter * max(n, 1), warmup * max(n,
            1), f'DataLoader ({n} workers, bs={self.per_gpu_batch_size})')
        del loader
