def first_order_update(model_output, sample):
    sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1
        ], scheduler.sigmas[scheduler.step_index]
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    h = lambda_t - lambda_s
    mu_xt = sigma_t / sigma_s * torch.exp(-h) * sample + alpha_t * (1 -
        torch.exp(-2.0 * h)) * model_output
    mu_xt = scheduler.dpm_solver_first_order_update(model_output=
        model_output, sample=sample, noise=torch.zeros_like(sample))
    sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
    if sigma > 0.0:
        noise = (prev_latents - mu_xt) / sigma
    else:
        noise = torch.tensor([0.0]).to(sample.device)
    prev_sample = mu_xt + sigma * noise
    return noise, prev_sample
