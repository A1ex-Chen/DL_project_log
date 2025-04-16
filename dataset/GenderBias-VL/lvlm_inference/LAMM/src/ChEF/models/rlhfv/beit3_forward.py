def forward(self, pixel_values, query_embed=None, encode_image=False,
    img_feat_layer=-1, attn_mask=None):
    assert (query_embed is not None) ^ encode_image
    B = pixel_values.size(0)
    dtype = self.beit3.vision_embed.proj.weight.dtype
    pixel_values = pixel_values.to(dtype)
    token_embeddings = self.beit3.vision_embed(pixel_values)
    multiway_split_position = -1
    if query_embed is not None:
        query_embed = torch.stack([query_embed] * B)
        multiway_split_position = token_embeddings.size(1)
        token_embeddings = torch.cat([token_embeddings, query_embed], dim=1)
    outputs = self.beit3.encoder(src_tokens=None, token_embeddings=
        token_embeddings, multiway_split_position=multiway_split_position,
        return_all_hiddens=encode_image, attn_mask=attn_mask)
    vision_hidden_states = outputs['encoder_out']
    if query_embed is not None:
        vision_hidden_states = vision_hidden_states[:, self.num_img_patches:]
    if encode_image:
        vision_hidden_states = outputs['encoder_states'][img_feat_layer][:,
            1:self.num_img_patches]
    return vision_hidden_states
