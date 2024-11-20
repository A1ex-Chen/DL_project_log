def second_order_update(model_output_list, sample):
    sigma_t, sigma_s0, sigma_s1 = scheduler.sigmas[scheduler.step_index + 1
        ], scheduler.sigmas[scheduler.step_index], scheduler.sigmas[
        scheduler.step_index - 1]
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = scheduler._sigma_to_alpha_sigma_t(sigma_s1)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    m0, m1 = model_output_list[-1], model_output_list[-2]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, 1.0 / r0 * (m0 - m1)
    mu_xt = sigma_t / sigma_s0 * torch.exp(-h) * sample + alpha_t * (1 -
        torch.exp(-2.0 * h)) * D0 + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))
        ) * D1
    sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
    if sigma > 0.0:
        noise = (prev_latents - mu_xt) / sigma
    else:
        noise = torch.tensor([0.0]).to(sample.device)
    prev_sample = mu_xt + sigma * noise
    return noise, prev_sample
