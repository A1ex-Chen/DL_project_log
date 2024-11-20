def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc
    =True, grad_attn_weight=True):
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-05
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2,
        keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply
    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight
    gradok = gradcheck(func, (value.double(), shapes, level_start_index,
        sampling_locations.double(), attention_weights.double(), im2col_step))
    print(f'* {gradok} check_gradient_numerical(D={channels})')
