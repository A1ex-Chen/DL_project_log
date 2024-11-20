def _train_speed(self, model_name: str, batch_size: int, sequence_length: int
    ) ->float:
    _train = self._prepare_train_func(model_name, batch_size, sequence_length)
    return self._measure_speed(_train)
