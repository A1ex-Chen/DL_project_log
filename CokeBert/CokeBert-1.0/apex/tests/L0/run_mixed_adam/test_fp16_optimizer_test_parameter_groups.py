def test_parameter_groups(self):
    ref_groups = [{'params': [self.ref_model.weight]}, {'params': [self.
        ref_model.bias]}]
    ref_optim = torch.optim.Adam(ref_groups)
    ref_optim = apex.fp16_utils.FP16_Optimizer(ref_optim, verbose=False)
    tst_groups = [{'params': [self.tst_model.weight]}, {'params': [self.
        tst_model.bias]}]
    tst_optim = apex.optimizers.FusedAdam(tst_groups)
    tst_optim = apex.optimizers.FP16_Optimizer(tst_optim)
    for i in range(self.iters):
        ref_loss = self.ref_model(self.x).sum()
        ref_optim.backward(ref_loss)
        ref_optim.step()
        tst_loss = self.tst_model(self.x).sum()
        tst_optim.backward(tst_loss)
        tst_optim.step()
        max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.
            parameters(), self.tst_model.parameters())
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
