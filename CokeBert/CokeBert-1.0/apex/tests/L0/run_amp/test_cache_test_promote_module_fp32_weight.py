def test_promote_module_fp32_weight(self):
    self.train_eval_train_test(PromoteModule, torch.float32)
