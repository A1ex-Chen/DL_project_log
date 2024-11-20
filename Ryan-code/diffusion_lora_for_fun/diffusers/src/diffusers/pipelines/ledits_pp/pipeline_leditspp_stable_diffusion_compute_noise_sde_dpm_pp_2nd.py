def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep,
    noise_pred, eta):

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
            torch.exp(-2.0 * h)) * D0 + 0.5 * (alpha_t * (1 - torch.exp(-
            2.0 * h))) * D1
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma
        else:
            noise = torch.tensor([0.0]).to(sample.device)
        prev_sample = mu_xt + sigma * noise
        return noise, prev_sample
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)
    model_output = scheduler.convert_model_output(model_output=noise_pred,
        sample=latents)
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output
    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, latents)
    else:
        noise, prev_sample = second_order_update(scheduler.model_outputs,
            latents)
    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1
    scheduler._step_index += 1
    return noise, prev_sample
