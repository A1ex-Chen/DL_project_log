def new_forward(self, hidden_states, pos_weight=pos_weight, neg_weight=
    neg_weight, **kwargs):
    cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
    batch_size, d_model = cond_hiddens.shape[:2]
    device, dtype = hidden_states.device, hidden_states.dtype
    weights = torch.ones(batch_size, d_model, device=device, dtype=dtype)
    out_pos = self.old_forward(hidden_states)
    out_neg = self.old_forward(hidden_states)
    if cached_pos_hiddens is not None:
        cached_pos_hs = cached_pos_hiddens.pop(0).to(hidden_states.device)
        cond_pos_hs = torch.cat([cond_hiddens, cached_pos_hs], dim=1)
        pos_weights = weights.clone().repeat(1, 1 + cached_pos_hs.shape[1] //
            d_model)
        pos_weights[:, d_model:] = pos_weight
        attn_with_weights = FabricCrossAttnProcessor()
        out_pos = attn_with_weights(self, cond_hiddens,
            encoder_hidden_states=cond_pos_hs, weights=pos_weights)
    else:
        out_pos = self.old_forward(cond_hiddens)
    if cached_neg_hiddens is not None:
        cached_neg_hs = cached_neg_hiddens.pop(0).to(hidden_states.device)
        uncond_neg_hs = torch.cat([uncond_hiddens, cached_neg_hs], dim=1)
        neg_weights = weights.clone().repeat(1, 1 + cached_neg_hs.shape[1] //
            d_model)
        neg_weights[:, d_model:] = neg_weight
        attn_with_weights = FabricCrossAttnProcessor()
        out_neg = attn_with_weights(self, uncond_hiddens,
            encoder_hidden_states=uncond_neg_hs, weights=neg_weights)
    else:
        out_neg = self.old_forward(uncond_hiddens)
    out = torch.cat([out_pos, out_neg], dim=0)
    return out
