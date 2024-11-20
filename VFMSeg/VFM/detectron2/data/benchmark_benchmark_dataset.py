def benchmark_dataset(self, num_iter, warmup=5):
    """
        Benchmark the speed of taking raw samples from the dataset.
        """

    def loader():
        while True:
            for k in self.sampler:
                yield self.dataset[k]
    self._benchmark(loader(), num_iter, warmup, 'Dataset Alone')
