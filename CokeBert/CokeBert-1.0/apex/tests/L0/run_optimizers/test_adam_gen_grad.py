def gen_grad(self, ref_param, tst_param):
    for p_ref, p_tst in zip(ref_param, tst_param):
        p_ref.grad = torch.rand_like(p_ref)
        p_tst.grad = p_ref.grad
