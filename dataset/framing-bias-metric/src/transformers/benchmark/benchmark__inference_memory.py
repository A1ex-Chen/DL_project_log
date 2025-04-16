def _inference_memory(self, model_name: str, batch_size: int,
    sequence_length: int) ->[Memory, Optional[MemorySummary]]:
    _inference = self._prepare_inference_func(model_name, batch_size,
        sequence_length)
    return self._measure_memory(_inference)
