def __init__(self, filename, image_folder=None, seed=None, **kwargs):
    super().__init__(**kwargs)
    self.filename = filename
    self.image_folder = image_folder
    self.rng = np.random.default_rng(seed)
    self.data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            self.data.append(line)
