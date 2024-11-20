def __getitem__(self, index):
    example = {}
    example['prompt'] = self.prompt
    example['index'] = index
    return example
