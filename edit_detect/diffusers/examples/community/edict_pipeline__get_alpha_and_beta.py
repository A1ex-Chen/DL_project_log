def _get_alpha_and_beta(self, t: torch.Tensor):
    t = int(t)
    alpha_prod = self.scheduler.alphas_cumprod[t
        ] if t >= 0 else self.scheduler.final_alpha_cumprod
    return alpha_prod, 1 - alpha_prod
