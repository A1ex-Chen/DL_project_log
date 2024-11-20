def measure_peak_usage_bytes(self):
    self._prepare_for_memory_profiling()
    model = self._model_provider()
    iteration = self._iteration_provider(model)
    iteration(*self._input_provider(batch_size=self._batch_size))
    torch.cuda.reset_max_memory_allocated()
    for _ in range(2):
        iteration(*self._input_provider(batch_size=self._batch_size))
    return torch.cuda.max_memory_allocated()
