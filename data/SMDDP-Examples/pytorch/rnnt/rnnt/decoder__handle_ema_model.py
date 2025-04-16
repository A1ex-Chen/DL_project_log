def _handle_ema_model(self, ema_model):
    if self.amp_level == 2:
        ema_model_eval = copy.deepcopy(ema_model)
        ema_model_eval.half()
    else:
        ema_model_eval = ema_model
    ema_model_eval.eval()
    return ema_model_eval
