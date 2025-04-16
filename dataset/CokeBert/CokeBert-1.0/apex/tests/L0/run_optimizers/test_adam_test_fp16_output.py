@unittest.skip('No longer support output fp16 param')
def test_fp16_output(self):
    nelem = 278011
    adam_option = {'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08,
        'weight_decay': 0, 'amsgrad': False}
    tensor = torch.rand(nelem, dtype=torch.float, device='cuda')
    ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([
        tensor], adam_option)
    fp16_param = torch.nn.Parameter(tensor.clone().half())
    for i in range(self.iters):
        half_grads = self.gen_mixed_grad(ref_param, tst_param)
        ref_optim.step()
        tst_optim.step(grads=half_grads, output_params=[fp16_param])
        max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
        max_abs_diff, max_rel_diff = self.get_max_diff(tst_param, [
            fp16_param.float()])
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
