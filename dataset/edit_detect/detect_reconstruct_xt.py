def reconstruct_xt(pipeline: DiffusionPipeline, xt: torch.Tensor, t: torch.
    Tensor) ->torch.Tensor:
    alpha_prod_t = pipeline.scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (xt - beta_prod_t ** 0.5 * ddpm_pred(pipeline,
        xt, t)) / alpha_prod_t ** 0.5
    return pred_original_sample
