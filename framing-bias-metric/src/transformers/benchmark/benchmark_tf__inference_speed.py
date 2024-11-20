def _inference_speed(self, model_name: str, batch_size: int,
    sequence_length: int) ->float:
    strategy = self.args.strategy
    assert strategy is not None, 'A device strategy has to be initialized before using TensorFlow.'
    _inference = self._prepare_inference_func(model_name, batch_size,
        sequence_length)
    return self._measure_speed(_inference)
