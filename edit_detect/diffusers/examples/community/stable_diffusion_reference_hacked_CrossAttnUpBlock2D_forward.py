def hacked_CrossAttnUpBlock2D_forward(self, hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...], temb: Optional[torch
    .Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None,
    cross_attention_kwargs: Optional[Dict[str, Any]]=None, upsample_size:
    Optional[int]=None, attention_mask: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
    eps = 1e-06
    for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        hidden_states = resnet(hidden_states, temb)
        hidden_states = attn(hidden_states, encoder_hidden_states=
            encoder_hidden_states, cross_attention_kwargs=
            cross_attention_kwargs, attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask, return_dict=False)[0
            ]
        if MODE == 'write':
            if gn_auto_machine_weight >= self.gn_weight:
                var, mean = torch.var_mean(hidden_states, dim=(2, 3),
                    keepdim=True, correction=0)
                self.mean_bank.append([mean])
                self.var_bank.append([var])
        if MODE == 'read':
            if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                var, mean = torch.var_mean(hidden_states, dim=(2, 3),
                    keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                mean_acc = sum(self.mean_bank[i]) / float(len(self.
                    mean_bank[i]))
                var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) +
                    eps) ** 0.5
                hidden_states_uc = (hidden_states - mean
                    ) / std * std_acc + mean_acc
                hidden_states_c = hidden_states_uc.clone()
                if do_classifier_free_guidance and style_fidelity > 0:
                    hidden_states_c[uc_mask] = hidden_states[uc_mask]
                hidden_states = style_fidelity * hidden_states_c + (1.0 -
                    style_fidelity) * hidden_states_uc
    if MODE == 'read':
        self.mean_bank = []
        self.var_bank = []
    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
    return hidden_states
