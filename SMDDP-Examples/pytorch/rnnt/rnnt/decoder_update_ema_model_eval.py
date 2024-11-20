def update_ema_model_eval(self, ema_model):
    ema_model_eval = self._handle_ema_model(ema_model)
    if type(self.batch_eval_mode) == str and self.batch_eval_mode.startswith(
        'cg'):
        if self.cg_captured == False:
            raise Exception(
                'CUDA graph for eval should be captured first before updating')
        else:
            overflow_buf = torch.cuda.IntTensor([0])
            amp_C.multi_tensor_scale(65536, overflow_buf, [list(
                ema_model_eval.parameters()), list(self.model.parameters())
                ], 1.0)
    else:
        self.model = ema_model_eval
