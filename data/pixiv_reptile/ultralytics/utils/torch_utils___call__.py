def __call__(self, epoch, fitness):
    """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
    if fitness is None:
        return False
    if fitness >= self.best_fitness:
        self.best_epoch = epoch
        self.best_fitness = fitness
    delta = epoch - self.best_epoch
    self.possible_stop = delta >= self.patience - 1
    stop = delta >= self.patience
    if stop:
        prefix = colorstr('EarlyStopping: ')
        LOGGER.info(
            f"""{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. Best results observed at epoch {self.best_epoch}, best model saved as best.pt.
To update EarlyStopping(patience={self.patience}) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping."""
            )
    return stop
