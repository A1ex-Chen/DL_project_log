def forward(self, hidden_states, timestep: Union[torch.Tensor, float, int],
    proj_embedding: torch.Tensor, encoder_hidden_states: Optional[torch.
    Tensor]=None, attention_mask: Optional[torch.BoolTensor]=None,
    return_dict: bool=True):
    """
        The [`PriorTransformer`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, embedding_dim)`):
                The currently predicted image embeddings.
            timestep (`torch.LongTensor`):
                Current denoising step.
            proj_embedding (`torch.Tensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
                If return_dict is True, a [`~models.prior_transformer.PriorTransformerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
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
    if self.embedding_proj_norm is not None:
        proj_embedding = self.embedding_proj_norm(proj_embedding)
    proj_embeddings = self.embedding_proj(proj_embedding)
    if (self.encoder_hidden_states_proj is not None and 
        encoder_hidden_states is not None):
        encoder_hidden_states = self.encoder_hidden_states_proj(
            encoder_hidden_states)
    elif self.encoder_hidden_states_proj is not None and encoder_hidden_states is None:
        raise ValueError(
            '`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set'
            )
    hidden_states = self.proj_in(hidden_states)
    positional_embeddings = self.positional_embedding.to(hidden_states.dtype)
    additional_embeds = []
    additional_embeddings_len = 0
    if encoder_hidden_states is not None:
        additional_embeds.append(encoder_hidden_states)
        additional_embeddings_len += encoder_hidden_states.shape[1]
    if len(proj_embeddings.shape) == 2:
        proj_embeddings = proj_embeddings[:, None, :]
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states[:, None, :]
    additional_embeds = additional_embeds + [proj_embeddings,
        time_embeddings[:, None, :], hidden_states]
    if self.prd_embedding is not None:
        prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(
            batch_size, -1, -1)
        additional_embeds.append(prd_embedding)
    hidden_states = torch.cat(additional_embeds, dim=1)
    additional_embeddings_len = (additional_embeddings_len +
        proj_embeddings.shape[1] + 1)
    if positional_embeddings.shape[1] < hidden_states.shape[1]:
        positional_embeddings = F.pad(positional_embeddings, (0, 0,
            additional_embeddings_len, self.prd_embedding.shape[1] if self.
            prd_embedding is not None else 0), value=0.0)
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
    if self.norm_in is not None:
        hidden_states = self.norm_in(hidden_states)
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask)
    hidden_states = self.norm_out(hidden_states)
    if self.prd_embedding is not None:
        hidden_states = hidden_states[:, -1]
    else:
        hidden_states = hidden_states[:, additional_embeddings_len:]
    predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)
    if not return_dict:
        return predicted_image_embedding,
    return PriorTransformerOutput(predicted_image_embedding=
        predicted_image_embedding)
