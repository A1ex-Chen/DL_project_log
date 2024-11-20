def test_grad_clip(self):
    ref_optim = torch.optim.Adam(self.ref_model.parameters())
    ref_optim = apex.fp16_utils.FP16_Optimizer(ref_optim, verbose=False)
    tst_optim = apex.optimizers.FusedAdam(self.tst_model.parameters(),
        max_grad_norm=0.01)
    tst_optim = apex.optimizers.FP16_Optimizer(tst_optim)
    for i in range(self.iters):
        ref_loss = self.ref_model(self.x).sum()
        ref_optim.backward(ref_loss)
        ref_optim.clip_master_grads(0.01)
        ref_optim.step()
        tst_loss = self.tst_model(self.x).sum()
        tst_optim.backward(tst_loss)
        tst_optim.step()
        max_abs_diff, max_rel_diff = self.get_max_diff(self.ref_model.
            parameters(), self.tst_model.parameters())
        self.assertLessEqual(max_abs_diff, self.max_abs_diff)
        self.assertLessEqual(max_rel_diff, self.max_rel_diff)
