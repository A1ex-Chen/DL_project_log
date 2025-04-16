def __post_init__(self):
    setattr(self, 'prompts', [i for i in range(self.num_classes)])
    setattr(self, 'name', f'{self.__naming_fn__()}')
