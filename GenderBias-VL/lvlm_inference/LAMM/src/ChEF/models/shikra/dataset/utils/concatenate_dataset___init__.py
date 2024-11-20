def __init__(self, cfgs, seed=42, portion=1):
    self.cfgs = cfgs
    self.seed = seed
    self.portion = portion
    dataset = TorchConcatDataset([DATASETS.build(cfg) for cfg in cfgs])
    target_len = int(len(dataset) * portion)
    indices = list(range(len(dataset))) * int(np.ceil(portion))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    indices = indices[:target_len]
    super().__init__(dataset, indices)
