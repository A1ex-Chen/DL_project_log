def test_adam_option(self):
    nelem = 1
    adam_option = {'lr': 0.01, 'betas': (0.6, 0.9), 'eps': 3e-06,
        'weight_decay': 0, 'amsgrad': False}
    tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
    ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([
        tensor], adam_option)
    for i in range(self.iters):
        self.gen_grad(ref_param, tst_param)
        ref_optim.step()
        tst_optim.step()
        max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
