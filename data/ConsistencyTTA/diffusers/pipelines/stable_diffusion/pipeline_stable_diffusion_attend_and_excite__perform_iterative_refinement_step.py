def _perform_iterative_refinement_step(self, latents: torch.Tensor, indices:
    List[int], loss: torch.Tensor, threshold: float, text_embeddings: torch
    .Tensor, step_size: float, t: int, max_refinement_steps: int=20):
    """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
    iteration = 0
    target_loss = max(0, 1.0 - threshold)
    while loss > target_loss:
        iteration += 1
        latents = latents.clone().detach().requires_grad_(True)
        self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()
        max_attention_per_index = (self.
            _aggregate_and_get_max_attention_per_token(indices=indices))
        loss = self._compute_loss(max_attention_per_index)
        if loss != 0:
            latents = self._update_latent(latents, loss, step_size)
        logger.info(f'\t Try {iteration}. loss: {loss}')
        if iteration >= max_refinement_steps:
            logger.info(
                f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                )
            break
    latents = latents.clone().detach().requires_grad_(True)
    _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
    self.unet.zero_grad()
    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
        indices=indices)
    loss = self._compute_loss(max_attention_per_index)
    logger.info(f'\t Finished with loss of: {loss}')
    return loss, latents, max_attention_per_index
