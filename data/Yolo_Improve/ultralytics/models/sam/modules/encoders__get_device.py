def _get_device(self) ->torch.device:
    """Returns the device of the first point embedding's weight tensor."""
    return self.point_embeddings[0].weight.device
