def _compute_logit(self, hidden, weight, bias, proj):
    if proj is None:
        logit = F.linear(hidden, weight, bias=bias)
    else:
        proj_hid = F.linear(hidden, proj.t().contiguous())
        logit = F.linear(proj_hid, weight, bias=bias)
    return logit
