def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for key, value in self.items():
        setattr(self, key, value)
    self.__frozen = True
