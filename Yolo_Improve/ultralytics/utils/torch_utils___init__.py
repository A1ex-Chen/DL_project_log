def __init__(self, patience=50):
    """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
    self.best_fitness = 0.0
    self.best_epoch = 0
    self.patience = patience or float('inf')
    self.possible_stop = False
