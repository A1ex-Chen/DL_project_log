def _ema_update(self, name: str, value: float, initial_value: float,
    momentum: float=0.9):
    """
        Apply EMA update to `self.name` using `value`.

        This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
        of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
        large variance and using it lead to lower performance. Therefore we maintain an EMA of
        #foreground to stabilize the normalizer.

        Args:
            name: name of the normalizer
            value: the new value to update
            initial_value: the initial value to start with
            momentum: momentum of EMA

        Returns:
            float: the updated EMA value
        """
    if hasattr(self, name):
        old = getattr(self, name)
    else:
        old = initial_value
    new = old * momentum + value * (1 - momentum)
    setattr(self, name, new)
    return new
