def forward(self, prev_output_tokens, self_attn_padding_mask=None,
    encoder_out=None, incremental_state=None, features_only=False,
    return_all_hiddens=False, token_embeddings=None, first_step=False, **kwargs
    ):
    x, _ = self.forward_embedding(prev_output_tokens, token_embeddings,
        incremental_state, first_step=first_step, **kwargs)
    x = x.transpose(0, 1)
    self_attn_rel_pos_bias = None
    slen = prev_output_tokens.size(1)
    if self.self_attn_relative_position is not None:
        self_attn_rel_pos_bias = self.self_attn_relative_position(batch_size
            =x.size(1), qlen=slen, klen=slen)
        if incremental_state is not None:
            self_attn_rel_pos_bias = self_attn_rel_pos_bias[:, -1:, :]
    cross_attn_rel_pos_bias = None
    if self.cross_attn_relative_position is not None:
        cross_attn_rel_pos_bias = self.cross_attn_relative_position(batch_size
            =x.size(1), qlen=slen, klen=encoder_out['encoder_out'].size(0))
        if incremental_state is not None:
            cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[:, -1:, :]
    self_attn_sope_rel_pos = None
    cross_attn_sope_rel_pos = None
    if self.cross_attn_sope is not None:
        cross_attn_sope_rel_pos = self.cross_attn_sope(slen + encoder_out[
            'encoder_out'].size(0))
    inner_states = [x]
    if encoder_out is None:
        l_aux = []
    else:
        l_aux = encoder_out['l_aux'] if 'l_aux' in encoder_out else []
    for idx, layer in enumerate(self.layers):
        if incremental_state is None or first_step:
            self_attn_mask = torch.triu(torch.zeros([x.size(0), x.size(0)])
                .float().fill_(float('-inf')).type_as(x), 1)
            if first_step and incremental_state is not None:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
        else:
            self_attn_mask = None
            if idx not in incremental_state:
                incremental_state[idx] = {}
        x, layer_attn, _, l_aux_i = layer(x, encoder_out['encoder_out'] if 
            encoder_out is not None else None, encoder_out[
            'encoder_padding_mask'] if encoder_out is not None else None, 
            incremental_state[idx] if incremental_state is not None else
            None, self_attn_mask=self_attn_mask, self_attn_padding_mask=
            self_attn_padding_mask, self_attn_rel_pos=
            self_attn_rel_pos_bias, cross_attn_rel_pos=
            cross_attn_rel_pos_bias, self_attn_sope_rel_pos=
            self_attn_sope_rel_pos, cross_attn_sope_rel_pos=
            cross_attn_sope_rel_pos)
        l_aux.append(l_aux_i)
        inner_states.append(x)
    if self.layer_norm is not None:
        x = self.layer_norm(x)
    x = x.transpose(0, 1)
    if not features_only:
        x = self.output_layer(x)
    return x, {'inner_states': inner_states, 'l_aux': l_aux, 'attn': [
        layer_attn.mean(dim=0) if layer_attn is not None else None]}
