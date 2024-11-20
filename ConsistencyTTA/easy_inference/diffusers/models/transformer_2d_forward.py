def forward(self, hidden_states: torch.Tensor, encoder_hidden_states:
    Optional[torch.Tensor]=None, timestep: Optional[torch.LongTensor]=None,
    class_labels: Optional[torch.LongTensor]=None, cross_attention_kwargs:
    Dict[str, Any]=None, attention_mask: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None, return_dict: bool=True
    ):
    """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class 
                labels conditioning.
            attention_mask ( `torch.Tensor` of shape (batch size, num latent pixels), *optional* ).
                Bias to add to attention scores.
            encoder_attention_mask ( `torch.Tensor` of shape (batch size, num encoder tokens), *optional* ).
                Bias to add to cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. 
            When returning a tuple, the first element is the sample tensor.
        """
    if self.is_input_continuous:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch,
                height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch,
                height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)
    elif self.is_input_vectorized:
        hidden_states = self.latent_image_embedding(hidden_states)
    elif self.is_input_patches:
        hidden_states = self.pos_embed(hidden_states)
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, timestep=
            timestep, cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels)
    if self.is_input_continuous:
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width,
                inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width,
                inner_dim).permute(0, 3, 1, 2).contiguous()
        output = hidden_states + residual
    elif self.is_input_vectorized:
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        logits = logits.permute(0, 2, 1)
        output = F.log_softmax(logits.double(), dim=1).float()
    elif self.is_input_patches:
        conditioning = self.transformer_blocks[0].norm1.emb(timestep,
            class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]
            ) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(shape=(-1, height, width,
            self.patch_size, self.patch_size, self.out_channels))
        hidden_states = torch.einsum('nhwpqc->nchpwq', hidden_states)
        output = hidden_states.reshape(shape=(-1, self.out_channels, height *
            self.patch_size, width * self.patch_size))
    if not return_dict:
        return output,
    return Transformer2DModelOutput(sample=output)
