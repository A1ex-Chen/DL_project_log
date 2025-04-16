def __len__(self):
    return sum([len(v) for v in self.model_zoo.values()])
