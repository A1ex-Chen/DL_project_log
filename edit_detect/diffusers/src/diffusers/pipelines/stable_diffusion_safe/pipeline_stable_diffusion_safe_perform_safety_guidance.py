def perform_safety_guidance(self, enable_safety_guidance, safety_momentum,
    noise_guidance, noise_pred_out, i, sld_guidance_scale, sld_warmup_steps,
    sld_threshold, sld_momentum_scale, sld_mom_beta):
    if enable_safety_guidance:
        if safety_momentum is None:
            safety_momentum = torch.zeros_like(noise_guidance)
        noise_pred_text, noise_pred_uncond = noise_pred_out[0], noise_pred_out[
            1]
        noise_pred_safety_concept = noise_pred_out[2]
        scale = torch.clamp(torch.abs(noise_pred_text -
            noise_pred_safety_concept) * sld_guidance_scale, max=1.0)
        safety_concept_scale = torch.where(noise_pred_text -
            noise_pred_safety_concept >= sld_threshold, torch.zeros_like(
            scale), scale)
        noise_guidance_safety = torch.mul(noise_pred_safety_concept -
            noise_pred_uncond, safety_concept_scale)
        noise_guidance_safety = (noise_guidance_safety + sld_momentum_scale *
            safety_momentum)
        safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta
            ) * noise_guidance_safety
        if i >= sld_warmup_steps:
            noise_guidance = noise_guidance - noise_guidance_safety
    return noise_guidance, safety_momentum
