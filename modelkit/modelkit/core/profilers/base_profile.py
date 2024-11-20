@contextmanager
def profile(self, *args, **kwargs) ->Generator:
    """Override this function for your custom profiler.

        Example:
            try:
                self.start(model_name)
                yield model_name
            finally:
                self.end(model_name)
        Usage:
            with self.profile('do something'):
                # do something
        """
    raise NotImplementedError
