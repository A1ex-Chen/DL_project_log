def test_whitelist_module_fp32_weight(self):
    self.train_eval_train_test(WhitelistModule, torch.float32)
