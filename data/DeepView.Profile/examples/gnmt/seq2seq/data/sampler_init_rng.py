def init_rng(self):
    """
        Creates new RNG, seed depends on current epoch idx.
        """
    rng = torch.Generator()
    seed = self.seeds[self.epoch]
    logging.info(f'Sampler for epoch {self.epoch} uses seed {seed}')
    rng.manual_seed(seed)
    return rng
