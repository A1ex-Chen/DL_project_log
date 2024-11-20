def benchmark_IPC(self, num_iter, warmup=10):
    """
        Benchmark the dataloader where each worker outputs nothing. This
        eliminates the IPC overhead compared to the regular dataloader.

        PyTorch multiprocessing's IPC only optimizes for torch tensors.
        Large numpy arrays or other data structure may incur large IPC overhead.
        """
    n = self.num_workers
    dataset = _EmptyMapDataset(MapDataset(self.dataset, self.mapper))
    loader = build_batch_data_loader(dataset, self.sampler, self.
        total_batch_size, num_workers=n)
    self._benchmark(iter(loader), num_iter * max(n, 1), warmup * max(n, 1),
        f'DataLoader ({n} workers, bs={self.per_gpu_batch_size}) w/o comm')
