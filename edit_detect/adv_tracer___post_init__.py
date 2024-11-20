def __post_init__(self):
    self.prompts = [i for i in range(self.num_classes)]
    self.batch_size = self.repeat
    self.num_warmup_steps = int(self.max_iter * 0.1)
    self.num_training_steps = self.max_iter
    self.name = f'{self.__naming_fn__()}'
