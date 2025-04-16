def unet_forward_with_cached_hidden_states(self, z_all, t, prompt_embd,
    cached_pos_hiddens: Optional[List[torch.Tensor]]=None,
    cached_neg_hiddens: Optional[List[torch.Tensor]]=None, pos_weights=(0.8,
    0.8), neg_weights=(0.5, 0.5)):
    if cached_pos_hiddens is None and cached_neg_hiddens is None:
        return self.unet(z_all, t, encoder_hidden_states=prompt_embd)
    local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.
        down_blocks) + 1)[:-1].tolist()
    local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.
        down_blocks) + 1)[:-1].tolist()
    for block, pos_weight, neg_weight in zip(self.unet.down_blocks + [self.
        unet.mid_block] + self.unet.up_blocks, local_pos_weights + [
        pos_weights[1]] + local_pos_weights[::-1], local_neg_weights + [
        neg_weights[1]] + local_neg_weights[::-1]):
        for module in block.modules():
            if isinstance(module, BasicTransformerBlock):

                def new_forward(self, hidden_states, pos_weight=pos_weight,
                    neg_weight=neg_weight, **kwargs):
                    cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0
                        )
                    batch_size, d_model = cond_hiddens.shape[:2]
                    device, dtype = hidden_states.device, hidden_states.dtype
                    weights = torch.ones(batch_size, d_model, device=device,
                        dtype=dtype)
                    out_pos = self.old_forward(hidden_states)
                    out_neg = self.old_forward(hidden_states)
                    if cached_pos_hiddens is not None:
                        cached_pos_hs = cached_pos_hiddens.pop(0).to(
                            hidden_states.device)
                        cond_pos_hs = torch.cat([cond_hiddens,
                            cached_pos_hs], dim=1)
                        pos_weights = weights.clone().repeat(1, 1 + 
                            cached_pos_hs.shape[1] // d_model)
                        pos_weights[:, d_model:] = pos_weight
                        attn_with_weights = FabricCrossAttnProcessor()
                        out_pos = attn_with_weights(self, cond_hiddens,
                            encoder_hidden_states=cond_pos_hs, weights=
                            pos_weights)
                    else:
                        out_pos = self.old_forward(cond_hiddens)
                    if cached_neg_hiddens is not None:
                        cached_neg_hs = cached_neg_hiddens.pop(0).to(
                            hidden_states.device)
                        uncond_neg_hs = torch.cat([uncond_hiddens,
                            cached_neg_hs], dim=1)
                        neg_weights = weights.clone().repeat(1, 1 + 
                            cached_neg_hs.shape[1] // d_model)
                        neg_weights[:, d_model:] = neg_weight
                        attn_with_weights = FabricCrossAttnProcessor()
                        out_neg = attn_with_weights(self, uncond_hiddens,
                            encoder_hidden_states=uncond_neg_hs, weights=
                            neg_weights)
                    else:
                        out_neg = self.old_forward(uncond_hiddens)
                    out = torch.cat([out_pos, out_neg], dim=0)
                    return out
                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)
    out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)
    for module in self.unet.modules():
        if isinstance(module, BasicTransformerBlock):
            module.attn1.forward = module.attn1.old_forward
            del module.attn1.old_forward
    return out
