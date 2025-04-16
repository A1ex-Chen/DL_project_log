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
        g_16, p_16, m_16 = [], [], []
        g_32, p_32, m_32 = [], [], []
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError(
                    'FusedNovoGrad does not support sparse gradients, please consider SparseAdam instead'
                    )
            state = self.state[p]
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p.data)
            if p.dtype == torch.float16:
                g_16.append(p.grad.data)
                p_16.append(p.data)
                m_16.append(state['exp_avg'])
            elif p.dtype == torch.float32:
                g_32.append(p.grad.data)
                p_32.append(p.data)
                m_32.append(state['exp_avg'])
            else:
                raise RuntimeError('FusedNovoGrad only support fp16 and fp32.')
        if 'exp_avg_sq' not in group:
            group['exp_avg_sq'] = [None, None]
            if group['init_zero']:
                group['exp_avg_sq'][0] = torch.cuda.FloatTensor(len(g_16)
                    ).contiguous().fill_(0)
                group['exp_avg_sq'][1] = torch.cuda.FloatTensor(len(g_32)
                    ).contiguous().fill_(0)
            else:
                if group['norm_type'] == 0:
                    v_16 = [torch.max(torch.abs(g.to(torch.float32))).item(
                        ) for g in g_16]
                    v_32 = [torch.max(torch.abs(g)).item() for g in g_32]
                elif group['norm_type'] == 2:
                    v_16 = [torch.sum(torch.pow(g.to(torch.float32), 2)).
                        sqrt().item() for g in g_16]
                    v_32 = [torch.sum(torch.pow(g, 2)).sqrt().item() for g in
                        g_32]
                else:
                    raise RuntimeError(
                        'FusedNovoGrad only support l2/inf norm now.')
                group['exp_avg_sq'][0] = torch.cuda.FloatTensor(v_16)
                group['exp_avg_sq'][1] = torch.cuda.FloatTensor(v_32)
        else:
            assert len(g_16) == group['exp_avg_sq'][0].numel()
            assert len(g_32) == group['exp_avg_sq'][1].numel()
        if len(g_16) > 0:
            multi_tensor_applier(self.multi_tensor_novograd, self.
                _dummy_overflow_buf, [g_16, p_16, m_16], group['exp_avg_sq'
                ][0], group['lr'], beta1, beta2, group['eps'], group['step'
                ], bias_correction, group['weight_decay'], grad_averaging,
                self.moment_mode, group['norm_type'])
        if len(g_32) > 0:
            multi_tensor_applier(self.multi_tensor_novograd, self.
                _dummy_overflow_buf, [g_32, p_32, m_32], group['exp_avg_sq'
                ][1], group['lr'], beta1, beta2, group['eps'], group['step'
                ], bias_correction, group['weight_decay'], grad_averaging,
                self.moment_mode, group['norm_type'])
    return loss
