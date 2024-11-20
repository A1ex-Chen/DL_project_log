def compute_dream_and_update_latents(unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin, timesteps: torch.Tensor, noise: torch.
    Tensor, noisy_latents: torch.Tensor, target: torch.Tensor,
    encoder_hidden_states: torch.Tensor, dream_detail_preservation: float=1.0
    ) ->Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[
        timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    dream_lambda = sqrt_one_minus_alphas_cumprod ** dream_detail_preservation
    pred = None
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    noisy_latents, target = None, None
    if noise_scheduler.config.prediction_type == 'epsilon':
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod *
            delta_noise)
        target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        raise NotImplementedError(
            'DREAM has not been implemented for v-prediction')
    else:
        raise ValueError(
            f'Unknown prediction type {noise_scheduler.config.prediction_type}'
            )
    return noisy_latents, target
