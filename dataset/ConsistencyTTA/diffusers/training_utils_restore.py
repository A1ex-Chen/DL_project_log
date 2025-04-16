def restore(self, parameters: Iterable[torch.nn.Parameter]) ->None:
    """
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
    if self.temp_stored_params is None:
        raise RuntimeError(
            'This ExponentialMovingAverage has no `store()`ed weights to `restore()`'
            )
    for c_param, param in zip(self.temp_stored_params, parameters):
        param.data.copy_(c_param.data)
    self.temp_stored_params = None
