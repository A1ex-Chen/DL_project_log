def compute_mask(self, t, default_mask):
    tensor_size = t.nelement()
    nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
    _validate_pruning_amount(nparams_toprune, tensor_size)
    mask = default_mask.clone(memory_format=torch.contiguous_format)
    if nparams_toprune != 0:
        t[t < 0] = 0
        indices = torch.nonzero(t == 0, as_tuple=True)
        mask.view(-1)[indices] = 0
    return mask
