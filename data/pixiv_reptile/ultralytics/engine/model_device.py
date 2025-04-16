@property
def device(self) ->torch.device:
    """
        Retrieves the device on which the model's parameters are allocated.

        This property is used to determine whether the model's parameters are on CPU or GPU. It only applies to models
        that are instances of nn.Module.

        Returns:
            (torch.device | None): The device (CPU/GPU) of the model if it is a PyTorch model, otherwise None.
        """
    return next(self.model.parameters()).device if isinstance(self.model,
        nn.Module) else None
