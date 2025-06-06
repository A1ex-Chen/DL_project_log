def __setattr__(self, name, value):
    if hasattr(self, '__frozen') and self.__frozen:
        raise Exception(
            f'You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.'
            )
    super().__setattr__(name, value)
