def gen_param_optim(self, tensors, adam_option):
    ref_param = []
    tst_param = []
    for tensor in tensors:
        ref_param.append(torch.nn.Parameter(tensor.clone()))
        tst_param.append(torch.nn.Parameter(tensor.clone()))
    ref_optim = torch.optim.Adam(ref_param, **adam_option)
    tst_optim = apex.optimizers.FusedAdam(tst_param, **adam_option)
    return ref_param, tst_param, ref_optim, tst_optim
