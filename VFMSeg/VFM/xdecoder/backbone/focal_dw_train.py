def train(self, mode=True):
    """Convert the model into training mode while keep layers freezed."""
    super(FocalNet, self).train(mode)
    self._freeze_stages()
