def print(self):
    """Print the confusion matrix to the console."""
    for i in range(self.nc + 1):
        LOGGER.info(' '.join(map(str, self.matrix[i])))
