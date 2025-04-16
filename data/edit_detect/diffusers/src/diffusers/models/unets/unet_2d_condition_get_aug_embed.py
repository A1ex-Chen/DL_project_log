def get_aug_embed(self, emb: torch.Tensor, encoder_hidden_states: torch.
    Tensor, added_cond_kwargs: Dict[str, Any]) ->Optional[torch.Tensor]:
    aug_emb = None
    if self.config.addition_embed_type == 'text':
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == 'text_image':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
        image_embs = added_cond_kwargs.get('image_embeds')
        text_embs = added_cond_kwargs.get('text_embeds', encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == 'text_time':
        if 'text_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
        text_embeds = added_cond_kwargs.get('text_embeds')
        if 'time_ids' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
        time_ids = added_cond_kwargs.get('time_ids')
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == 'image':
        if 'image_embeds' not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
        image_embs = added_cond_kwargs.get('image_embeds')
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == 'image_hint':
        if ('image_embeds' not in added_cond_kwargs or 'hint' not in
            added_cond_kwargs):
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
        image_embs = added_cond_kwargs.get('image_embeds')
        hint = added_cond_kwargs.get('hint')
        aug_emb = self.add_embedding(image_embs, hint)
    return aug_emb
