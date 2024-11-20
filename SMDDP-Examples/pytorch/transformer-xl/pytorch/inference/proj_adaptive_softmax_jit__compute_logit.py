def _compute_logit(self, hidden, weight, bias, proj: Optional[torch.Tensor]):
    if proj is None:
        logit = F.linear(hidden, weight, bias=bias)
    else:
        logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
        if bias is not None:
            logit = logit + bias
    return logit
