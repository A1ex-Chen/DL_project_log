def copy_to(self, parameters: Iterable[torch.nn.Parameter]) ->None:
    """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
    parameters = list(parameters)
    for s_param, param in zip(self.shadow_params, parameters):
        param.data.copy_(s_param.data)
