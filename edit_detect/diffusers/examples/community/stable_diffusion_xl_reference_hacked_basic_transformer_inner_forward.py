def hacked_basic_transformer_inner_forward(self, hidden_states: torch.
    Tensor, attention_mask: Optional[torch.Tensor]=None,
    encoder_hidden_states: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None, timestep: Optional
    [torch.LongTensor]=None, cross_attention_kwargs: Dict[str, Any]=None,
    class_labels: Optional[torch.LongTensor]=None):
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self
            .norm1(hidden_states, timestep, class_labels, hidden_dtype=
            hidden_states.dtype))
    else:
        norm_hidden_states = self.norm1(hidden_states)
    cross_attention_kwargs = (cross_attention_kwargs if 
        cross_attention_kwargs is not None else {})
    if self.only_cross_attention:
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=
            encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs)
    else:
        if MODE == 'write':
            self.bank.append(norm_hidden_states.detach().clone())
            attn_output = self.attn1(norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.
                only_cross_attention else None, attention_mask=
                attention_mask, **cross_attention_kwargs)
        if MODE == 'read':
            if attention_auto_machine_weight > self.attn_weight:
                attn_output_uc = self.attn1(norm_hidden_states,
                    encoder_hidden_states=torch.cat([norm_hidden_states] +
                    self.bank, dim=1), **cross_attention_kwargs)
                attn_output_c = attn_output_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    attn_output_c[uc_mask] = self.attn1(norm_hidden_states[
                        uc_mask], encoder_hidden_states=norm_hidden_states[
                        uc_mask], **cross_attention_kwargs)
                attn_output = style_fidelity * attn_output_c + (1.0 -
                    style_fidelity) * attn_output_uc
                self.bank.clear()
            else:
                attn_output = self.attn1(norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.
                    only_cross_attention else None, attention_mask=
                    attention_mask, **cross_attention_kwargs)
    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states
    if self.attn2 is not None:
        norm_hidden_states = self.norm2(hidden_states, timestep
            ) if self.use_ada_layer_norm else self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=
            encoder_hidden_states, attention_mask=encoder_attention_mask,
            **cross_attention_kwargs)
        hidden_states = attn_output + hidden_states
    norm_hidden_states = self.norm3(hidden_states)
    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]
            ) + shift_mlp[:, None]
    ff_output = self.ff(norm_hidden_states)
    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = ff_output + hidden_states
    return hidden_states
