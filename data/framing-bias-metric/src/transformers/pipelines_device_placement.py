@contextmanager
def device_placement(self):
    """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
    if self.framework == 'tf':
        with tf.device('/CPU:0' if self.device == -1 else '/device:GPU:{}'.
            format(self.device)):
            yield
    else:
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        yield
