def update(self, *args, **kwargs):
    raise Exception(
        f'You cannot use ``update`` on a {self.__class__.__name__} instance.')
