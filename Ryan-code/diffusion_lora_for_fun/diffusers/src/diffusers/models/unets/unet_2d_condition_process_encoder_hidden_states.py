def process_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Dict[str, Any]) ->torch.Tensor:
    if (self.encoder_hid_proj is not None and self.config.
        encoder_hid_dim_type == 'text_proj'):
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == 'text_image_proj':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
        image_embeds = added_cond_kwargs.get('image_embeds')
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states,
            image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == 'image_proj':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
        image_embeds = added_cond_kwargs.get('image_embeds')
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == 'ip_image_proj':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
        image_embeds = added_cond_kwargs.get('image_embeds')
        image_embeds = self.encoder_hid_proj(image_embeds)
        encoder_hidden_states = encoder_hidden_states, image_embeds
    return encoder_hidden_states
