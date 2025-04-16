def apply_multi_tensor_ema(model_weight_list, ema_model_weight_list, decay,
    overflow_buf):
    if not decay:
        return
    amp_C.multi_tensor_axpby(65536, overflow_buf, [ema_model_weight_list,
        model_weight_list, ema_model_weight_list], decay, 1 - decay, -1)
