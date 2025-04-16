def forward(self, ctx_embeddings: torch.Tensor, ctx_begin_pos: list,
    input_ids: Optional[torch.LongTensor]=None, position_ids: Optional[
    torch.LongTensor]=None, inputs_embeds: Optional[torch.Tensor]=None
    ) ->torch.Tensor:
    if ctx_embeddings is None:
        ctx_len = 0
    else:
        ctx_len = ctx_embeddings.shape[1]
    seq_length = (input_ids.shape[-1] if input_ids is not None else
        inputs_embeds.shape[-2]) + ctx_len
    if position_ids is None:
        position_ids = self.position_ids[:, :seq_length]
    if inputs_embeds is None:
        inputs_embeds = self.token_embedding(input_ids)
        input_embeds_ctx = []
        bsz = inputs_embeds.shape[0]
        if ctx_embeddings is not None:
            for i in range(bsz):
                cbp = ctx_begin_pos[i]
                prefix = inputs_embeds[i, :cbp]
                suffix = inputs_embeds[i, cbp:]
                input_embeds_ctx.append(torch.cat([prefix, ctx_embeddings[i
                    ], suffix], dim=0))
            inputs_embeds = torch.stack(input_embeds_ctx, dim=0)
    position_embeddings = self.position_embedding(position_ids)
    embeddings = inputs_embeds + position_embeddings
    return embeddings
