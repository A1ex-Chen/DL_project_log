def __init__(self, global_args):
    super().__init__()
    self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers=4)
    self.diffusion_ema = deepcopy(self.diffusion)
    self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
