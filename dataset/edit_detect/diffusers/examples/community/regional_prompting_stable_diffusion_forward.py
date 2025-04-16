def forward(hidden_states: torch.Tensor, encoder_hidden_states: Optional[
    torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, temb:
    Optional[torch.Tensor]=None, scale: float=1.0) ->torch.Tensor:
    attn = module
    xshape = hidden_states.shape
    self.hw = h, w = split_dims(xshape[1], *orig_hw)
    if revers:
        nx, px = hidden_states.chunk(2)
    else:
        px, nx = hidden_states.chunk(2)
    if equal:
        hidden_states = torch.cat([px for i in range(regions)] + [nx for i in
            range(regions)], 0)
        encoder_hidden_states = torch.cat([conds] + [unconds])
    else:
        hidden_states = torch.cat([px for i in range(regions)] + [nx], 0)
        encoder_hidden_states = torch.cat([conds] + [unconds])
    residual = hidden_states
    args = () if USE_PEFT_BACKEND else (scale,)
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width
            ).transpose(1, 2)
    batch_size, sequence_length, _ = (hidden_states.shape if 
        encoder_hidden_states is None else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask,
            sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1,
            attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)
            ).transpose(1, 2)
    args = () if USE_PEFT_BACKEND else (scale,)
    query = attn.to_q(hidden_states, *args)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states, *args)
    value = attn.to_v(encoder_hidden_states, *args)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    hidden_states = scaled_dot_product_attention(self, query, key, value,
        attn_mask=attention_mask, dropout_p=0.0, is_causal=False, getattn=
        'PRO' in mode)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, 
        attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.to_out[0](hidden_states, *args)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size,
            channel, height, width)
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / attn.rescale_output_factor
    if any(x in mode for x in ['COL', 'ROW']):
        reshaped = hidden_states.reshape(hidden_states.size()[0], h, w,
            hidden_states.size()[2])
        center = reshaped.shape[0] // 2
        px = reshaped[0:center] if equal else reshaped[0:-batch]
        nx = reshaped[center:] if equal else reshaped[-batch:]
        outs = [px, nx] if equal else [px]
        for out in outs:
            c = 0
            for i, ocell in enumerate(ocells):
                for icell in icells[i]:
                    if 'ROW' in mode:
                        out[0:batch, int(h * ocell[0]):int(h * ocell[1]),
                            int(w * icell[0]):int(w * icell[1]), :] = out[c *
                            batch:(c + 1) * batch, int(h * ocell[0]):int(h *
                            ocell[1]), int(w * icell[0]):int(w * icell[1]), :]
                    else:
                        out[0:batch, int(h * icell[0]):int(h * icell[1]),
                            int(w * ocell[0]):int(w * ocell[1]), :] = out[c *
                            batch:(c + 1) * batch, int(h * icell[0]):int(h *
                            icell[1]), int(w * ocell[0]):int(w * ocell[1]), :]
                    c += 1
        px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0:batch], nx)
        hidden_states = torch.cat([nx, px], 0) if revers else torch.cat([px,
            nx], 0)
        hidden_states = hidden_states.reshape(xshape)
    elif 'PRO' in mode:
        px, nx = torch.chunk(hidden_states) if equal else hidden_states[0:-
            batch], hidden_states[-batch:]
        if (h, w) in self.attnmasks and self.maskready:

            def mask(input):
                out = torch.multiply(input, self.attnmasks[h, w])
                for b in range(batch):
                    for r in range(1, regions):
                        out[b] = out[b] + out[r * batch + b]
                return out
            px, nx = (mask(px), mask(nx)) if equal else (mask(px), nx)
        px, nx = (px[0:batch], nx[0:batch]) if equal else (px[0:batch], nx)
        hidden_states = torch.cat([nx, px], 0) if revers else torch.cat([px,
            nx], 0)
    return hidden_states
