def hacked_mid_forward(self, *args, **kwargs):
    eps = 1e-06
    x = self.original_forward(*args, **kwargs)
    if MODE == 'write':
        if gn_auto_machine_weight >= self.gn_weight:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True,
                correction=0)
            self.mean_bank.append(mean)
            self.var_bank.append(var)
    if MODE == 'read':
        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True,
                correction=0)
            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
            mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
            var_acc = sum(self.var_bank) / float(len(self.var_bank))
            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps
                ) ** 0.5
            x_uc = (x - mean) / std * std_acc + mean_acc
            x_c = x_uc.clone()
            if do_classifier_free_guidance and style_fidelity > 0:
                x_c[uc_mask] = x[uc_mask]
            x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
        self.mean_bank = []
        self.var_bank = []
    return x
