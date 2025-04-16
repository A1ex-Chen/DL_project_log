def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
    half_grads = []
    for p_ref, p_tst in zip(ref_param, tst_param):
        half_grads.append(torch.rand_like(p_ref).half())
        p_ref.grad = half_grads[-1].float() / scale
    return half_grads
