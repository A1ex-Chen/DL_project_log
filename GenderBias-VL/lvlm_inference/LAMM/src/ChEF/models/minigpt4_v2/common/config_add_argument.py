def add_argument(self, *args, **kwargs):
    """
        Assume the first argument is the name of the argument.
        """
    self.arguments[args[0]] = self._Argument(*args, **kwargs)
