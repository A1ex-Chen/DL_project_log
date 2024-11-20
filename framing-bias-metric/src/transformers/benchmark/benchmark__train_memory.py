def _train_memory(self, model_name: str, batch_size: int, sequence_length: int
    ) ->[Memory, Optional[MemorySummary]]:
    _train = self._prepare_train_func(model_name, batch_size, sequence_length)
    return self._measure_memory(_train)
