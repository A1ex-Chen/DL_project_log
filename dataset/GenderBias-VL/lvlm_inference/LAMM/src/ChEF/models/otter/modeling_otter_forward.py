def forward(self, vision_x: torch.Tensor, lang_x: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.
    Tensor]=None, use_cached_vision_x: bool=False, clear_conditioned_layers:
    bool=True, past_key_values: Optional[torch.Tensor]=None, use_cache:
    bool=False, **kwargs) ->CausalLMOutputWithPast:
    """
        Forward pass of Otter.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
    assert vision_x is not None or use_cached_vision_x, 'Must provide either vision_x or use_cached_vision_x to True.'
    if use_cached_vision_x:
        assert vision_x is None, 'Expect vision_x to be None when use_cached_vision_x is True.'
        assert self.lang_encoder.is_conditioned()
    else:
        self._encode_vision_x(vision_x=vision_x)
    output = self.lang_encoder(input_ids=lang_x, attention_mask=
        attention_mask, labels=labels, past_key_values=past_key_values,
        use_cache=use_cache, **kwargs)
    if clear_conditioned_layers:
        self.lang_encoder.clear_conditioned_layers()
    return output
