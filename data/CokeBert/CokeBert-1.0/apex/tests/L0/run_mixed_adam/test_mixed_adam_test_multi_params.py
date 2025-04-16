def test_multi_params(self):
    sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
    adam_option = {'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08,
        'weight_decay': 0, 'amsgrad': False}
    tensors = []
    for size in sizes:
        tensors.append(torch.rand(size, dtype=torch.float, device='cuda'))
    ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors,
        adam_option)
    for i in range(self.iters):
        half_grads = self.gen_mixed_grad(ref_param, tst_param)
        ref_optim.step()
        tst_optim.step(grads=half_grads)
        max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
