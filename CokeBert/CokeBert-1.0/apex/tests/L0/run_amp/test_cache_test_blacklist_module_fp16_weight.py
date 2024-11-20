def test_blacklist_module_fp16_weight(self):
    self.train_eval_train_test(BlacklistModule, torch.float16)
