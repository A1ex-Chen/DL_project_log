def test_promote_module_fp16_weight(self):
    self.train_eval_train_test(PromoteModule, torch.float16)
