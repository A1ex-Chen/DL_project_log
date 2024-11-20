def __delitem__(self, *args, **kwargs):
    raise Exception(
        f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.'
        )
