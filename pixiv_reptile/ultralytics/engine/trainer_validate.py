def validate(self):
    """
        Runs validation on create_self_data set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
    metrics = self.validator(self)
    fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())
    if not self.best_fitness or self.best_fitness < fitness:
        self.best_fitness = fitness
    return metrics, fitness
