def get_xt_by_t(pipeline, x0: torch.Tensor, t: int, noise: torch.Tensor):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(x0.device)
    alpha_prod_t = alphas_cumprod[t].reshape((-1, *([1] * len(x0.shape[1:]))))
    beta_prod_t = 1 - alpha_prod_t
    return alpha_prod_t * x0 + beta_prod_t * noise
