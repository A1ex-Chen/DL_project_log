def hacked_DownBlock2D_forward(self, hidden_states: torch.Tensor, temb:
    Optional[torch.Tensor]=None, **kwargs: Any) ->Tuple[torch.Tensor, ...]:
    eps = 1e-06
    output_states = ()
    for i, resnet in enumerate(self.resnets):
        hidden_states = resnet(hidden_states, temb)
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
        output_states = output_states + (hidden_states,)
    if MODE == 'read':
        self.mean_bank = []
        self.var_bank = []
    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        output_states = output_states + (hidden_states,)
    return hidden_states, output_states
