def to(self, device):
    """ Puts the model to the device.

        Args:
            device (device): pytorch device
        """
    model = super().to(device)
    model._device = device
    return model
