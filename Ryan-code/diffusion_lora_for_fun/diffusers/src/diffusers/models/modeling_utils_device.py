@property
def device(self) ->torch.device:
    """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
    return get_parameter_device(self)
