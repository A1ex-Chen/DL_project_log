def step(self, closure=None):
    """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
    loss = None
    if closure is not None:
        loss = closure()
    for group in self.param_groups:
        bias_correction = 1 if group['bias_correction'] else 0
        beta1, beta2 = group['betas']
        grad_averaging = 1 if group['grad_averaging'] else 0
        if 'step' in group:
            group['step'] += 1
        else:
            group['step'] = 1
        g_16, p_16, m_16, v_16 = [], [], [], []
        g_32, p_32, m_32, v_32 = [], [], [], []
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError(
                    'FusedLAMB does not support sparse gradients, please consider SparseAdam instead'
                    )
            state = self.state[p]
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            if p.dtype == torch.float16:
                g_16.append(p.grad.data)
                p_16.append(p.data)
                m_16.append(state['exp_avg'])
                v_16.append(state['exp_avg_sq'])
            elif p.dtype == torch.float32:
                g_32.append(p.grad.data)
                p_32.append(p.data)
                m_32.append(state['exp_avg'])
                v_32.append(state['exp_avg_sq'])
            else:
                raise RuntimeError('FusedLAMB only support fp16 and fp32.')
        if len(g_16) > 0:
            multi_tensor_applier(self.multi_tensor_lamb, self.
                _dummy_overflow_buf, [g_16, p_16, m_16, v_16], group['lr'],
                beta1, beta2, group['eps'], group['step'], bias_correction,
                group['weight_decay'], grad_averaging, self.adam_w_mode,
                group['max_grad_norm'])
        if len(g_32) > 0:
            multi_tensor_applier(self.multi_tensor_lamb, self.
                _dummy_overflow_buf, [g_32, p_32, m_32, v_32], group['lr'],
                beta1, beta2, group['eps'], group['step'], bias_correction,
                group['weight_decay'], grad_averaging, self.adam_w_mode,
                group['max_grad_norm'])
    return loss
