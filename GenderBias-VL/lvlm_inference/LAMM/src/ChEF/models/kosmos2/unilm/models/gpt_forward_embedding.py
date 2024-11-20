def forward_embedding(self, tokens, token_embedding=None, incremental_state
    =None, first_step: bool=False, mlm_features: Optional[Tensor]=None,
    gpt_input_mask: Optional[Tensor]=None, img_features: Optional[Tensor]=
    None, img_gpt_input_mask: Optional[Tensor]=None, aud_features: Optional
    [Tensor]=None, aud_gpt_input_mask: Optional[Tensor]=None, chunk_tokens:
    Optional[Tensor]=None, segment_tokens: Optional[Tensor]=None):
    positions = None
    if self.embed_positions is not None:
        positions = self.embed_positions(tokens, incremental_state=
            incremental_state)
        if self.chunk_emb is not None:
            chunk_emb = self.chunk_emb(chunk_tokens)
            positions += chunk_emb
        if self.segment_emb is not None:
            segment_emb = self.segment_emb(segment_tokens)
            positions += segment_emb
    if incremental_state is not None and not first_step:
        tokens = tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]
    if token_embedding is None:
        token_embedding = self.embed_tokens(tokens)
    gpt_embed_output = token_embedding
    if mlm_features is not None:
        gpt_embed_output[gpt_input_mask] = mlm_features
    if img_features is not None:
        gpt_embed_output[img_gpt_input_mask] = img_features
    if aud_features is not None:
        gpt_embed_output[aud_gpt_input_mask] = aud_features
    x = embed = self.embed_scale * gpt_embed_output
    if positions is not None:
        x += positions
    if self.layernorm_embedding is not None:
        x = self.layernorm_embedding(x)
    x = self.dropout_module(x)
    return x, embed
