def benchmark_distributed(self, num_iter, warmup=10):
    """
        Benchmark the dataloader in each distributed worker, and log results of
        all workers. This helps understand the final performance as well as
        the variances among workers.

        It also prints startup time (first iter) of the dataloader.
        """
    gpu = comm.get_world_size()
    dataset = MapDataset(self.dataset, self.mapper)
    n = self.num_workers
    loader = build_batch_data_loader(dataset, self.sampler, self.
        total_batch_size, num_workers=n)
    timer = Timer()
    loader = iter(loader)
    next(loader)
    startup_time = timer.seconds()
    logger.info('Dataloader startup time: {:.2f} seconds'.format(startup_time))
    comm.synchronize()
    avg, all_times = self._benchmark(loader, num_iter * max(n, 1), warmup *
        max(n, 1))
    del loader
    self._log_time(
        f'DataLoader ({gpu} GPUs x {n} workers, total bs={self.total_batch_size})'
        , avg, all_times, True)
