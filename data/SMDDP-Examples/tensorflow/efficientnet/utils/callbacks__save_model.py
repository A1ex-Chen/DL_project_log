def _save_model(self, epoch, logs):
    assert isinstance(self.model.optimizer, optimizer_factory.MovingAverage)
    if self.update_weights:
        self.model.optimizer.assign_average_vars(self.model.variables)
        return super()._save_model(epoch, logs)
    else:
        non_avg_weights = self.model.get_weights()
        self.model.optimizer.assign_average_vars(self.model.variables)
        result = super()._save_model(epoch, logs)
        self.model.set_weights(non_avg_weights)
        return result
