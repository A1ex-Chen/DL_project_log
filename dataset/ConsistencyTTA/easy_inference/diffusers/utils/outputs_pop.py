def pop(self, *args, **kwargs):
    raise Exception(
        f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')
