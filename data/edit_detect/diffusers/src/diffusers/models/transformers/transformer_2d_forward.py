def forward(self, hidden_states: torch.Tensor, encoder_hidden_states:
    Optional[torch.Tensor]=None, timestep: Optional[torch.LongTensor]=None,
    added_cond_kwargs: Dict[str, torch.Tensor]=None, class_labels: Optional
    [torch.LongTensor]=None, cross_attention_kwargs: Dict[str, Any]=None,
    attention_mask: Optional[torch.Tensor]=None, encoder_attention_mask:
    Optional[torch.Tensor]=None, return_dict: bool=True):
    """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get('scale', None) is not None:
            logger.warning(
                'Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.'
                )
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)
            ) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(
            hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    if self.is_input_continuous:
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states, inner_dim = self._operate_on_continuous_inputs(
            hidden_states)
    elif self.is_input_vectorized:
        hidden_states = self.latent_image_embedding(hidden_states)
    elif self.is_input_patches:
        height, width = hidden_states.shape[-2
            ] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        (hidden_states, encoder_hidden_states, timestep, embedded_timestep) = (
            self._operate_on_patched_inputs(hidden_states,
            encoder_hidden_states, timestep, added_cond_kwargs))
    for block in self.transformer_blocks:
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):

                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block), hidden_states, attention_mask,
                encoder_hidden_states, encoder_attention_mask, timestep,
                cross_attention_kwargs, class_labels, **ckpt_kwargs)
        else:
            hidden_states = block(hidden_states, attention_mask=
                attention_mask, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask, timestep=
                timestep, cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels)
    if self.is_input_continuous:
        output = self._get_output_for_continuous_inputs(hidden_states=
            hidden_states, residual=residual, batch_size=batch_size, height
            =height, width=width, inner_dim=inner_dim)
    elif self.is_input_vectorized:
        output = self._get_output_for_vectorized_inputs(hidden_states)
    elif self.is_input_patches:
        output = self._get_output_for_patched_inputs(hidden_states=
            hidden_states, timestep=timestep, class_labels=class_labels,
            embedded_timestep=embedded_timestep, height=height, width=width)
    if not return_dict:
        return output,
    return Transformer2DModelOutput(sample=output)
