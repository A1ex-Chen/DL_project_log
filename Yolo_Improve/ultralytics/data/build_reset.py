def reset(self):
    """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
    self.iterator = self._get_iterator()
