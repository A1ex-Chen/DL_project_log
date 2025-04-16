def _inference_memory(self, model_name: str, batch_size: int,
    sequence_length: int) ->[Memory, Optional[MemorySummary]]:
    if self.args.is_gpu:
        tf.config.experimental.set_memory_growth(self.args.gpu_list[self.
            args.device_idx], True)
    strategy = self.args.strategy
    assert strategy is not None, 'A device strategy has to be initialized before using TensorFlow.'
    _inference = self._prepare_inference_func(model_name, batch_size,
        sequence_length)
    return self._measure_memory(_inference)
