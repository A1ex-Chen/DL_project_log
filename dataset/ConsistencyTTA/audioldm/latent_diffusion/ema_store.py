def store(self, parameters):
    """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
    self.collected_params = [param.clone() for param in parameters]
