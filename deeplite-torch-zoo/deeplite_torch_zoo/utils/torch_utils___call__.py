def __call__(self, epoch, fitness):
    if fitness >= self.best_fitness:
        self.best_epoch = epoch
        self.best_fitness = fitness
    delta = epoch - self.best_epoch
    self.possible_stop = delta >= self.patience - 1
    stop = delta >= self.patience
    if stop:
        LOGGER.info(
            f'EarlyStopping patience {self.patience} exceeded, stopping training.'
            )
    return stop
