def forward(self, hidden_states, timestep: Union[torch.Tensor, float, int],
    proj_embedding: torch.FloatTensor, encoder_hidden_states: torch.
    FloatTensor, attention_mask: Optional[torch.BoolTensor]=None,
    return_dict: bool=True):
    """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`torch.long`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
            [`~models.prior_transformer.PriorTransformerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
    batch_size = hidden_states.shape[0]
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=
            hidden_states.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(hidden_states.device)
    timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype,
        device=timesteps.device)
    timesteps_projected = self.time_proj(timesteps)
    timesteps_projected = timesteps_projected.to(dtype=self.dtype)
    time_embeddings = self.time_embedding(timesteps_projected)
    proj_embeddings = self.embedding_proj(proj_embedding)
    encoder_hidden_states = self.encoder_hidden_states_proj(
        encoder_hidden_states)
    hidden_states = self.proj_in(hidden_states)
    prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(
        batch_size, -1, -1)
    positional_embeddings = self.positional_embedding.to(hidden_states.dtype)
    hidden_states = torch.cat([encoder_hidden_states, proj_embeddings[:,
        None, :], time_embeddings[:, None, :], hidden_states[:, None, :],
        prd_embedding], dim=1)
    hidden_states = hidden_states + positional_embeddings
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)
            ) * -10000.0
        attention_mask = F.pad(attention_mask, (0, self.
            additional_embeddings), value=0.0)
        attention_mask = (attention_mask[:, None, :] + self.
            causal_attention_mask).to(hidden_states.dtype)
        attention_mask = attention_mask.repeat_interleave(self.config.
            num_attention_heads, dim=0)
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask)
    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states[:, -1]
    predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)
    if not return_dict:
        return predicted_image_embedding,
    return PriorTransformerOutput(predicted_image_embedding=
        predicted_image_embedding)
