def setdefault(self, *args, **kwargs):
    raise Exception(
        f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.'
        )
