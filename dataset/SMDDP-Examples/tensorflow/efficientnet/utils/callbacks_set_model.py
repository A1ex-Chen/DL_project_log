def set_model(self, model):
    if not isinstance(model.optimizer, optimizer_factory.MovingAverage):
        raise TypeError(
            'AverageModelCheckpoint is only used when trainingwith MovingAverage'
            )
    return super().set_model(model)
