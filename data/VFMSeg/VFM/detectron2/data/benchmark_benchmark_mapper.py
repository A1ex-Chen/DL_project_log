def benchmark_mapper(self, num_iter, warmup=5):
    """
        Benchmark the speed of taking raw samples from the dataset and map
        them in a single process.
        """

    def loader():
        while True:
            for k in self.sampler:
                yield self.mapper(self.dataset[k])
    self._benchmark(loader(), num_iter, warmup,
        'Single Process Mapper (sec/sample)')
