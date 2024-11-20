def forward(self, latent_image_embeds: torch.Tensor, image_embeds: torch.
    Tensor, prompt_embeds: torch.Tensor, timestep_img: Union[torch.Tensor,
    float, int], timestep_text: Union[torch.Tensor, float, int], data_type:
    Optional[Union[torch.Tensor, float, int]]=1, encoder_hidden_states=None,
    cross_attention_kwargs=None):
    """
        Args:
            latent_image_embeds (`torch.Tensor` of shape `(batch size, latent channels, height, width)`):
                Latent image representation from the VAE encoder.
            image_embeds (`torch.Tensor` of shape `(batch size, 1, clip_img_dim)`):
                CLIP-embedded image representation (unsqueezed in the first dimension).
            prompt_embeds (`torch.Tensor` of shape `(batch size, seq_len, text_dim)`):
                CLIP-embedded text representation.
            timestep_img (`torch.long` or `float` or `int`):
                Current denoising step for the image.
            timestep_text (`torch.long` or `float` or `int`):
                Current denoising step for the text.
            data_type: (`torch.int` or `float` or `int`, *optional*, defaults to `1`):
                Only used in UniDiffuser-v1-style models. Can be either `1`, to use weights trained on nonpublic data,
                or `0` otherwise.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            cross_attention_kwargs (*optional*):
                Keyword arguments to supply to the cross attention layers, if used.


        Returns:
            `tuple`: Returns relevant parts of the model's noise prediction: the first element of the tuple is tbe VAE
            image embedding, the second element is the CLIP image embedding, and the third element is the CLIP text
            embedding.
        """
    batch_size = latent_image_embeds.shape[0]
    vae_hidden_states = self.vae_img_in(latent_image_embeds)
    clip_hidden_states = self.clip_img_in(image_embeds)
    text_hidden_states = self.text_in(prompt_embeds)
    num_text_tokens, num_img_tokens = text_hidden_states.size(1
        ), vae_hidden_states.size(1)
    if not torch.is_tensor(timestep_img):
        timestep_img = torch.tensor([timestep_img], dtype=torch.long,
            device=vae_hidden_states.device)
    timestep_img = timestep_img * torch.ones(batch_size, dtype=timestep_img
        .dtype, device=timestep_img.device)
    timestep_img_token = self.timestep_img_proj(timestep_img)
    timestep_img_token = timestep_img_token.to(dtype=self.dtype)
    timestep_img_token = self.timestep_img_embed(timestep_img_token)
    timestep_img_token = timestep_img_token.unsqueeze(dim=1)
    if not torch.is_tensor(timestep_text):
        timestep_text = torch.tensor([timestep_text], dtype=torch.long,
            device=vae_hidden_states.device)
    timestep_text = timestep_text * torch.ones(batch_size, dtype=
        timestep_text.dtype, device=timestep_text.device)
    timestep_text_token = self.timestep_text_proj(timestep_text)
    timestep_text_token = timestep_text_token.to(dtype=self.dtype)
    timestep_text_token = self.timestep_text_embed(timestep_text_token)
    timestep_text_token = timestep_text_token.unsqueeze(dim=1)
    if self.use_data_type_embedding:
        assert data_type is not None, 'data_type must be supplied if the model uses a data type embedding'
        if not torch.is_tensor(data_type):
            data_type = torch.tensor([data_type], dtype=torch.int, device=
                vae_hidden_states.device)
        data_type = data_type * torch.ones(batch_size, dtype=data_type.
            dtype, device=data_type.device)
        data_type_token = self.data_type_token_embedding(data_type).unsqueeze(
            dim=1)
        hidden_states = torch.cat([timestep_img_token, timestep_text_token,
            data_type_token, text_hidden_states, clip_hidden_states,
            vae_hidden_states], dim=1)
    else:
        hidden_states = torch.cat([timestep_img_token, timestep_text_token,
            text_hidden_states, clip_hidden_states, vae_hidden_states], dim=1)
    if self.use_data_type_embedding:
        pos_embed = torch.cat([self.pos_embed[:, :1 + 1, :], self.
            data_type_pos_embed_token, self.pos_embed[:, 1 + 1:, :]], dim=1)
    else:
        pos_embed = self.pos_embed
    hidden_states = hidden_states + pos_embed
    hidden_states = self.pos_embed_drop(hidden_states)
    hidden_states = self.transformer(hidden_states, encoder_hidden_states=
        encoder_hidden_states, timestep=None, class_labels=None,
        cross_attention_kwargs=cross_attention_kwargs, return_dict=False,
        hidden_states_is_embedding=True, unpatchify=False)[0]
    if self.use_data_type_embedding:
        (t_img_token_out, t_text_token_out, data_type_token_out, text_out,
            img_clip_out, img_vae_out) = (hidden_states.split((1, 1, 1,
            num_text_tokens, 1, num_img_tokens), dim=1))
    else:
        (t_img_token_out, t_text_token_out, text_out, img_clip_out, img_vae_out
            ) = (hidden_states.split((1, 1, num_text_tokens, 1,
            num_img_tokens), dim=1))
    img_vae_out = self.vae_img_out(img_vae_out)
    height = width = int(img_vae_out.shape[1] ** 0.5)
    img_vae_out = img_vae_out.reshape(shape=(-1, height, width, self.
        patch_size, self.patch_size, self.out_channels))
    img_vae_out = torch.einsum('nhwpqc->nchpwq', img_vae_out)
    img_vae_out = img_vae_out.reshape(shape=(-1, self.out_channels, height *
        self.patch_size, width * self.patch_size))
    img_clip_out = self.clip_img_out(img_clip_out)
    text_out = self.text_out(text_out)
    return img_vae_out, img_clip_out, text_out
