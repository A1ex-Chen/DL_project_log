@unittest.skip('No longer support fuse scaling')
def test_scale(self):
    nelem = 278011
    adam_option = {'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08,
        'weight_decay': 0, 'amsgrad': False}
    tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
    ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([
        tensor], adam_option)
    for i in range(self.iters):
        scale = random.random() * 1000
        half_grads = self.gen_mixed_grad(ref_param, tst_param, scale)
        ref_optim.step()
        tst_optim.step(grads=half_grads, scale=scale)
        max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
