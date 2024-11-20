def step(self):
    with torch.no_grad():
        weight_decays = []
        for group in self.optim.param_groups:
            weight_decay = group['weight_decay'
                ] if 'weight_decay' in group else 0
            weight_decays.append(weight_decay)
            group['weight_decay'] = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)
                if param_norm != 0 and grad_norm != 0:
                    adaptive_lr = self.trust_coefficient * param_norm / (
                        grad_norm + param_norm * weight_decay + self.eps)
                    if self.clip:
                        adaptive_lr = min(adaptive_lr / group['lr'], 1)
                    p.grad.data += weight_decay * p.data
                    p.grad.data *= adaptive_lr
    self.optim.step()
    for i, group in enumerate(self.optim.param_groups):
        group['weight_decay'] = weight_decays[i]
