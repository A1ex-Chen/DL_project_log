@staticmethod
def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) ->torch.Tensor:
    """Update the latent according to the computed loss."""
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents],
        retain_graph=True)[0]
    latents = latents - step_size * grad_cond
    return latents
