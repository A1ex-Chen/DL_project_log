def put_scalars(self, *, smoothing_hint=True, **kwargs):
    """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
    for k, v in kwargs.items():
        self.put_scalar(k, v, smoothing_hint=smoothing_hint)
